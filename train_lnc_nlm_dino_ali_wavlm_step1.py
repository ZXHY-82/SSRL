import os, time, random, argparse, numpy as np, sys
import torch, torch.nn as nn, torch.nn.functional as F
import librosa, scipy.io.wavfile as sciwav, torchaudio
from scipy.signal import fftconvolve
import torch.multiprocessing as mp, torch.distributed as dist, torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
from dataset import WavBatchDistributedSampler
from utils.utils import AverageMeter, ProgressMeter, get_lr, accuracy, str2bool
from utils.spk_veri_metric import  SVevaluation
from python_speech_features import sigproc
import sklearn.metrics as skmetrics
import scipy.optimize as optimize
from statistics import mode
from sklearn.mixture import GaussianMixture
from modules.ecapa_tdnn_dino_ali import ECAPA_TDNN_WavLM as ECAPA_TDNN
from modules.back_classifier import ArcFace
from torchaudio import transforms
from torch.nn import Parameter
import math
from wavlm.WavLM import WavLM, WavLMConfig

parser = argparse.ArgumentParser(description='label correction and noisy label modeling')
parser.add_argument('--save_dir', default='test', type=str)
# model setups
parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--feat_dim', default=1024, type=int)
parser.add_argument('--embd_dim', default=512, type=int)
parser.add_argument('--ema_decay', default=[0.999, 0.9999], nargs='+', type=float)
parser.add_argument('--use_sinkhorn', default=False, type=str2bool)
parser.add_argument('--ignore_removed_label_in_teacher', default=True, type=str2bool)
parser.add_argument('--noisy_label_mnodeling', default=True, type=str2bool)
parser.add_argument('--arcface', default=True, type=str2bool)
parser.add_argument('--layer_nums', default=10, type=int)
# data augmentation
parser.add_argument('--crop_dur', default=[2, 6], nargs='+', type=float)
parser.add_argument('--snr_range', default=[0, 20], nargs='+', type=int)
# training dataset
parser.add_argument('--data_name', default='dev_vox2_dino_ali_k8000', type=str)
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('-b', '--batch_size', default=256, type=int)
# test dataset
parser.add_argument('--val_data_name', default='vox_test', type=str)
parser.add_argument('--val_freq', default=1, type=int)
# eer and cost
parser.add_argument('--ptar', default=[0.01], nargs='+', type=float)
# optimizer
parser.add_argument('--wd', '--weight_decay', default=5e-5, type=float)
# learning rate scheduler
parser.add_argument('--lr', default=0.001, type=float)
# others
parser.add_argument('--laebl_update_freq', default=1, type=int)
parser.add_argument('--laebl_update_queue_len', default=5, type=int)
parser.add_argument('--start_epoch_step1', default=0, type=int)
parser.add_argument('--warmup_epoch_step1', default=5, type=int)
parser.add_argument('--fix_epoch_step1', default=25, type=int)
# single node distributed parallel training
parser.add_argument('--seed', default=3007123, type=int)
parser.add_argument('-p', '--print-freq', default=10, type=int)
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--port', default='8758', type=str)
parser.add_argument('--gpu', default='4,5,6,7', type=str)

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.system('mkdir -p exp/%s' % args.save_dir)
    os.system('cp %s exp/%s' % (sys.argv[0], args.save_dir))

    args.ngpus = torch.cuda.device_count()
    args.distributed = args.ngpus > 1
    if args.distributed:
        mp.spawn(main_worker, nprocs=args.ngpus, args=(args,))
    else:
        main_worker(0, args)


def main_worker(gpu, args):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    
    args.rank = gpu
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method='tcp://127.0.0.1:%s' % args.port, world_size=args.ngpus, rank=args.rank)
        args.batch_size = int(args.batch_size / args.ngpus)
        args.workers = int((args.workers + args.ngpus - 1) / args.ngpus)

    utt2wav = [line.split() for line in open('data/%s/wav.scp' % args.data_name)]

    utt2spk = {line.split()[0]:int(line.split()[1]) for line in open('data/%s/utt2spk' % args.data_name)}

    noise_list = {'noise': [i.strip('\n') for i in open('data/envir/noise_wav_list')],
                  'music': [i.strip('\n') for i in open('data/envir/music_wav_list')],
                  'babb': [i.strip('\n') for i in open('data/envir/speech_wav_list')],
                  'reverb': [i.strip('\n') for i in open('data/envir/simu_rir_list')]}

    trn_dataset = WavDataset(utt2wav, utt2spk, args.fs,
                             is_aug=True, snr=args.snr_range, noise_list=noise_list, crop_dur=args.crop_dur,
                             laebl_update_queue_len=args.laebl_update_queue_len)
    tst_dataset = WavDataset([line.split() for line in open('data/%s/wav.scp' % args.val_data_name)], fs=args.fs)
    if args.distributed:
        trn_sampler = WavBatchDistributedSampler(trn_dataset, args.batch_size, shuffle=True)
        tst_sampler = WavBatchDistributedSampler(tst_dataset, batch_size=1, shuffle=False)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.workers, pin_memory=True, sampler=trn_sampler)
        tst_loader = DataLoader(tst_dataset, batch_size=1, drop_last=False, num_workers=args.workers, pin_memory=True, sampler=tst_sampler)
    else:
        trn_loader = DataLoader(trn_dataset, num_workers=args.workers, pin_memory=True, shuffle=True, batch_size=args.batch_size, drop_last=True)
        tst_loader = DataLoader(tst_dataset, num_workers=args.workers, pin_memory=True, shuffle=False, batch_size=1)

    # EER & cost calculator
    val_utt = [line.split()[0] for line in open('data/%s/wav.scp' % args.val_data_name)]
    if args.gpu == 0:
        eer_cal = SVevaluation('data/%s/trials' % args.val_data_name, val_utt, ptar=args.ptar)


    # create model
    num_spks = len(set(utt2spk.values()))
    model = SpeakerModel(feat_dim=args.feat_dim, embd_dim=args.embd_dim, dropout=0.5, num_spks=num_spks, arcface=args.arcface, layer_nums=args.layer_nums)

    model_ptm = WavLM_Large(sub_layers=args.layer_nums)

    if not args.arcface:
        model.classifier.bias.data.zero_()

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_ptm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ptm)
        model_ptm.cuda(args.gpu)
        model_ptm = torch.nn.parallel.DistributedDataParallel(model_ptm, device_ids=[args.gpu])
    else:
        model = model.cuda(args.gpu)
        model_ptm = model_ptm.cuda(args.gpu)
    
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm_step1_epochs = args.warmup_epoch_step1
    fix_step1_epochs = args.fix_epoch_step1
    total_step1_epochs = warm_step1_epochs + fix_step1_epochs
    scheduler = LR_Scheduler(optimizer, warm_step1_epochs,
                             args.lr, args.lr/100, fix_step1_epochs,
                             args.lr/2, 0, len(trn_loader))


    # optionally resume from a checkpoint
    if args.start_epoch_step1 != 0:
        checkpoint = torch.load('exp/%s/model_step1_%d.pkl' % (args.save_dir, args.start_epoch_step1-1), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ptm.load_state_dict(checkpoint['model_ptm'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.iter = args.start_epoch_step1 * len(trn_loader)
        checkpoint = torch.load('exp/%s/random_state.pkl' % (args.save_dir), map_location='cpu')
        random.setstate(checkpoint['random'])
        np.random.set_state(checkpoint['np'])
        torch.set_rng_state(checkpoint['torch'])
    elif args.rank == 0:
        print(str(model), flush=True)

    trn_loader.dataset.aug_rate = 0.66
    for epoch in range(args.start_epoch_step1, total_step1_epochs):
        if args.distributed:
            trn_sampler.set_epoch(epoch)

        loss, prec = train_step1(trn_loader, model, model_ptm, criterion, optimizer, scheduler, epoch, args)
        results = 'Epoch %3d  LR %.5f  ' % (epoch, scheduler.get_lr()) + \
                  'Loss %s  ' % (loss) + \
                  'Accuracy %s  ' % (prec)

        embd = test(tst_loader, model_ptm, model, args)[:len(val_utt)]

        if args.rank == 0:
            eer_cal.update_embd(embd)
            eer, cost = eer_cal.eer_cost()
            results += 'EER %2.4f  ' % (eer) + \
                       'DCF %.4f' % (cost)
              
        if args.rank == 0:
            torch.save({'model': model.state_dict(),
                        'model_ptm': model_ptm.state_dict(),
                        'optimizer': optimizer.state_dict(),
                       }, 'exp/%s/model_step1_%d.pkl' % (args.save_dir, epoch))
            torch.save({'random': random.getstate(),
                        'np': np.random.get_state(),
                        'torch': torch.get_rng_state()
                       }, 'exp/%s/random_state.pkl' % args.save_dir)
            print('Validate ' + results, flush=True)

def train_step1(trn_loader, model, model_ptm, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':3.2f')
    progress = ProgressMeter(len(trn_loader), [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    model.train()
    model_ptm.eval()

    indices_to_ignore = torch.LongTensor(trn_loader.dataset.label_to_remove).cuda(args.gpu)

    end = time.time()
    for i, (feats, _, label, _, _) in enumerate(trn_loader):

        data_time.update(time.time() - end)

        label = label.cuda(args.gpu)

        with torch.no_grad():
            feats = model_ptm(feats.cuda(args.gpu))
    
        outputs = model(feats, label=label)
        outputs[:, indices_to_ignore] = -float('inf')

        loss = criterion(outputs, label).mean()
        losses.update(loss.item(), feats.size(1))
        top1.update(accuracy(outputs, label)[0].item(), feats.size(1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, suffix='LR %.6f'  % get_lr(optimizer))


    if args.distributed:
        loss = torch.tensor(losses.avg).cuda(args.gpu)
        prec1 = torch.tensor(top1.avg).cuda(args.gpu)
        dist.all_reduce(loss, op=dist.reduce_op.SUM)
        dist.all_reduce(prec1, op=dist.reduce_op.SUM)
        return '%.4f' % (loss/args.ngpus), '%3.2f' % (prec1/args.ngpus)
    else:
        return '%.4f' % (losses.avg), '%3.2f' % (top1.avg)


def test(tst_loader, model_ptm, model, args):
    model.eval()
    model_ptm.eval()
    
    with torch.no_grad():
        for j, (feats, _) in enumerate(tst_loader):
            if len(feats[0]) > 16000*100:
                feats = feats[:,:16000*100]
            feats = model_ptm(feats.cuda(args.gpu))
            embd = model(feats, encoder_output=True)
            all_embd = embd if j == 0 else torch.cat((all_embd, embd))
                
    if args.distributed:
        gather_embd = [torch.zeros_like(all_embd).cuda(args.gpu) for i in range(dist.get_world_size())]
        dist.all_gather(gather_embd, all_embd)
        all_embd = torch.cat(gather_embd)

    return all_embd.cpu().numpy()


class EMAteacher(nn.Module):
    def __init__(self, model,
                 ema_decay=[0.996, 0.9999], ema_anneal_end_step=None):
        '''
        Args:
            model: the EMA teacher
            ema_decay: as form of a list [initial ema decay rate, final ema decay rate]
            ema_anneal_end_step: when to finish annealing ema decay rate
            min_target_var: stop training if target var falls below this
            min_pred_var: stop training if prediction var falls below this
        '''
        super().__init__()
        self.ema_model = model
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.ema_decay_start = ema_decay[0]
        self.ema_decay_end = ema_decay[1]
        self.ema_anneal_end_step = ema_anneal_end_step

        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

    def set_decay(self):
        curr_step = int(self.num_updates)

        if curr_step >= self.ema_anneal_end_step:
            decay = self.ema_decay_end
        else:
            r = self.ema_decay_end - self.ema_decay_start
            pct_remaining = 1 - curr_step / self.ema_anneal_end_step
            decay = self.ema_decay_end - r * pct_remaining

        self.ema_decay = decay
        self.num_updates += 1

    @torch.no_grad()
    def ema_step(self, new_model):
        self.set_decay()
        if self.ema_decay >= 1:
            return

        if dist.is_initialized():
            new_model = new_model.module

        ema_state_dict = self.ema_model.state_dict()
        ema_params = self.ema_model.state_dict()

        for key, param in new_model.named_parameters():
            ema_param = ema_params[key]
            if not param.requires_grad:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(self.ema_decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.ema_decay)
            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            if 'num_batches_tracked' in key:
                continue
            ema_param = ema_params[key]
            ema_param.mul_(self.ema_decay)
            ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.ema_decay)
            ema_state_dict[key] = ema_param

        self.ema_model.load_state_dict(ema_state_dict)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) \
               + ', easy_margin = ' + str(False) + ')'


class SpeakerModel(nn.Module):
    def __init__(self, feat_dim=80, embd_dim=128, dropout=0, num_spks=6000, arcface=False, layer_nums=10):
        super(SpeakerModel, self).__init__()
        self.encoder = ECAPA_TDNN(input_size=feat_dim, lin_neurons=embd_dim, channels=[1024, 1024, 1024, 1024, 3072],  log_input=False)
        self.dropout = nn.Dropout(p=dropout) if (dropout > 0 and arcface is False) else None
        self.feature_weight = nn.Parameter(torch.zeros(layer_nums))
        self.arcface = arcface
        if arcface:
            self.classifier = ArcFace(embd_dim, num_spks, s=32.0, m = 0.2)
        else:
            self.classifier = nn.Linear(embd_dim, num_spks)
        
    def forward(self, x, label=None, encoder_output=False):
        norm_weights = F.softmax(self.feature_weight, dim=-1)
        x = torch.tensordot(norm_weights, x, dims=([0], [0]))
        x = x.permute(0, 2, 1) # B,D,T
        x = self.encoder(x)
        if encoder_output:
            return x
        if self.dropout is not None:
            x = self.dropout(x)
        if self.arcface:
            return self.classifier(x, label)
        else:
            return self.classifier(x)


class WavLM_Large(nn.Module):
    def __init__(self, sub_layers=10):
        super(WavLM_Large, self).__init__()
        ptm_ckpt = torch.load('./wavlm/WavLM-Large.pt')
        ckpt_model = ptm_ckpt['model']
        ckpt_cfg = ptm_ckpt['cfg']
        self.sub_layers = sub_layers
        ckpt_cfg['encoder_layers'] = self.sub_layers
        sub_ptm = {}
        for i, j in ckpt_model.items():
            if 'encoder.layers.' in i:
                tmp = i.split('.')
                if int(tmp[2]) >= self.sub_layers:
                    continue
            sub_ptm[i] = j
        cfg = WavLMConfig(ckpt_cfg)
        self.model_ptm = WavLM(cfg)
        self.model_ptm.load_state_dict(sub_ptm)
    
    def forward(self, x):
        _, layer_results = self.model_ptm.extract_features(x, output_layer=self.sub_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        layer_reps = layer_reps[1:]
        x = torch.stack(layer_reps)
        return x


class WavDataset(Dataset):
    def __init__(self, wav_scp, utt2label=None, fs=16000, is_aug=False, snr=None, noise_list=None, crop_dur=None, laebl_update_queue_len=5):
        self.wav_scp = wav_scp
        self.utt2label = utt2label
        self.fs = fs
        self.clean_label_prob = np.ones(len(self.wav_scp), dtype='float32')
        if utt2label is not None:
            initial_queue_size = laebl_update_queue_len // 2
            self.utt2spk_evolve = None if utt2label is None else [[utt2label[u]]*initial_queue_size for u, _ in self.wav_scp]
            self.laebl_update_queue_len = laebl_update_queue_len
            self.original_labels = set(utt2label.values())
            self.label_to_remove = []

        self.is_aug=is_aug
        self.snr = snr
        self.noise_list = noise_list
        self.preemph = 0.97
        self.aug_rate = 1.0

        self.crop_dur = crop_dur

    def __len__(self):
        return len(self.wav_scp)

    def _load_data(self, filename):
        if os.path.splitext(filename)[-1] == '.wav':
            fs, signal = sciwav.read(filename, mmap=True)
        elif os.path.splitext(filename)[-1] == '.m4a':
            try:
                signal, fs = librosa.load(filename, sr=self.fs)
            except:
                print('FileError', filename)
                signal, fs = librosa.load(filename.replace('DATA1', 'NASdata/AudioData'), sr=self.fs)
        elif os.path.splitext(filename)[-1] == '.flac':
            signal, fs = librosa.load(filename, sr=self.fs)
        if fs != self.fs:
            effect = [['rate', str(self.fs)]]
            signal, _ = torchaudio.sox_effects.apply_effects_tensor(torch.tensor(signal.astype('float32').reshape(1, -1)), self.fs, effect)
            signal = signal.numpy()[0]
        return signal
    
    def _norm_speech(self, signal):
        if np.abs(signal).max() == 0:
            return signal
        signal = signal / (np.abs(signal).max())
        return signal
    
    def _augmentation(self, signal):
        signal = self._norm_speech(signal)
        noise_types = random.choice(['reverb', 'noise', 'none'])
        
        if noise_types == 'reverb':
            power = (signal ** 2).mean()
            rir = self._load_data(random.choice(self.noise_list[noise_types]))
            rir = (rir - min(rir)) / (max(rir) - min(rir))
            max_ind = np.argmax(np.abs(rir))
            rir = rir[max_ind:]
            signal = fftconvolve(signal, rir)[:signal.shape[0]]
            power2 = (signal ** 2).mean()
            signal = np.sqrt(power / max(power2, 1e-10)) * signal
            return signal
        else:
            noise_signal = np.zeros(signal.shape[0], dtype='float32')
            for noise_type in random.choice([['noise'], ['music'], ['babb', 'music'], ['babb'] * random.randint(3, 8)]):
                noise = self._load_data(random.choice(self.noise_list[noise_type]))
                noise = self._truncate_speech(noise, signal.shape[0])
                noise_signal = noise_signal + self._norm_speech(noise)
            snr = random.uniform(self.snr[0], self.snr[1])
            power = (signal ** 2).mean()
            noise_power = (noise_signal ** 2).mean()
            sigma_n = (
                10 ** (-snr / 20)
                * np.sqrt(power)
                / np.sqrt(max(noise_power, 1e-10))
            )
            return signal + noise_signal * sigma_n

    def _truncate_speech(self, signal, tlen, offset=None):
        if tlen == None:
            return signal
        if signal.shape[0] <= tlen:
            signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
        if offset is None:
            offset = random.randint(0, signal.shape[0] - tlen)
        return np.array(signal[offset : offset + tlen])

    def infer_new_labels(self, new_labels, is_update):
        if not is_update:
            return [label for _, label in new_labels]
        infered_labels = []
        for idx, label in new_labels:
            try:
                label_queue = self.utt2spk_evolve[idx] + [label]
                infered_labels.append(mode(label_queue[-self.laebl_update_queue_len:]))
            except:
                infered_labels.append(label)
        return infered_labels
    
    def update_labels(self, new_labels=None, update=True):
        if new_labels is not None:
            for idx, label in new_labels:
                self.utt2spk_evolve[idx].append(label)
                self.utt2spk_evolve[idx] = self.utt2spk_evolve[idx][-self.laebl_update_queue_len:]

        utt2label = {}
        for idx in range(len(self.wav_scp)):
            utt = self.wav_scp[idx][0]
            try:
                utt2label[utt] = mode(self.utt2spk_evolve[idx])
            except:
                utt2label[utt] = self.utt2spk_evolve[idx][-1]
        all_label = [utt2label[i] for i in utt2label]

        if update:
            self.utt2label = utt2label
            self.label_to_remove = list(self.original_labels - set(all_label))

        return nmi_acc_pur(all_label) + '  %d' % len(set(all_label))

    def update_clean_label_probability(self, teacher_loss):
        idx = np.array([i for i, _ in teacher_loss])
        loss_tr = np.array([l for _, l in teacher_loss])

        loss_tr = np.log(loss_tr+1e-12)
        max_perc = np.percentile(loss_tr, 99.9)
        min_perc = max(np.percentile(loss_tr, 0.1), -20)
        keep_idx = (loss_tr<=max_perc) & (loss_tr>=min_perc)
        gmm = GaussianMixture(n_components=2, max_iter=500, tol=1e-5).fit(loss_tr[keep_idx].reshape(-1, 1))
        
        loss_tr[loss_tr>=max_perc] = max_perc
        loss_tr[loss_tr<=min_perc] = min_perc
        p_clean = gmm.predict_proba(loss_tr.reshape(-1, 1))
        p_clean = p_clean[:, np.argmin(gmm.means_)] / p_clean.sum(axis=1)

        self.clean_label_prob[idx] = p_clean

        lt = [l.split()[0].split('-')[0] for l in open('data/vox2dev/utt2spk')]
        l_map = {l:i for i, l in enumerate(set(lt))}
        lt = [l_map[l] for l in lt]

        lp = [self.utt2label[i] for i in self.utt2label]
        l_map = {l:i for i, l in enumerate(set(lp))}
        lp = [l_map[l] for l in lp]

        lp, lt = map_lp(lp, lt)

        clean_label_pred = self.clean_label_prob > 0.5
        return f'TP: {(clean_label_pred & (lp==lt)).sum() / len(lt):.4f}  ' + \
               f'FP: {(clean_label_pred & (lp!=lt)).sum() / len(lt):.4f}  ' + \
               f'TN: {((~clean_label_pred) & (lp!=lt)).sum() / len(lt):.4f}  ' + \
               f'FN: {((~clean_label_pred) & (lp==lt)).sum() / len(lt):.4f}  '
            
    def __getitem__(self, idx):
        utt, wav = self.wav_scp[idx]
        label = self.utt2label[utt] if self.utt2label else utt
        wav = self._load_data(wav)

        if not self.is_aug:
            wav = self._norm_speech(wav)
            wav = sigproc.preemphasis(wav, self.preemph)
            wav = torch.from_numpy(wav.astype('float32'))
            return wav, label

        short_wav = self._truncate_speech(wav, int(self.crop_dur[0] * self.fs))
        if random.random() <= self.aug_rate:
            short_wav = self._augmentation(short_wav)
        short_wav = self._norm_speech(short_wav)
        short_wav = sigproc.preemphasis(short_wav, self.preemph)
        short_wav = torch.from_numpy(short_wav.astype('float32'))

        long_wav = self._truncate_speech(wav, int(self.crop_dur[1] * self.fs))
        long_wav = self._norm_speech(long_wav)
        long_wav = sigproc.preemphasis(long_wav, self.preemph)
        long_wav = torch.from_numpy(long_wav.astype('float32'))
        
        return short_wav, long_wav, label, self.clean_label_prob[idx], idx


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs,
                 base_lr, final_lr, label_fixed_epochs,
                 laebl_update_lr, laebl_update_epochs, iter_per_epoch):
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lrs = np.linspace(0.0, base_lr, warmup_iter+1)[1:]

        fixed_iter = label_fixed_epochs * iter_per_epoch
        fixed_lrs = (np.cos(np.pi * np.linspace(0, 1, fixed_iter+1)[1:]) + 1) / 2 * (base_lr - final_lr) + final_lr
                
        update_iter = laebl_update_epochs * iter_per_epoch
        update_lrs = (np.cos(np.pi * np.linspace(0, 1, update_iter+1)[1:]) + 1) / 2 * (laebl_update_lr - final_lr) + final_lr

        self.lr_schedule = np.concatenate((warmup_lrs, fixed_lrs, update_lrs))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for g in self.optimizer.param_groups:
            lr = g['lr'] = self.lr_schedule[self.iter]      
        self.iter += 1
        self.current_lr = lr
        return lr
    
    def get_lr(self):
        return self.current_lr
    

def Sinkhorn(out, max_iter=3, eps=0.05):
    Q = torch.exp(out / eps).t() # Q is K-by-B for consistency with notations from our paper
    K = Q.shape[0] # how many prototypes
    B = Q.shape[1] # number of samples to assign

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(max_iter):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if dist.is_initialized():
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor


def nmi_acc_pur(lp):
    lt = [l.split()[0].split('-')[0] for l in open('data/vox2dev/utt2spk')]
    l_map = {l:i for i, l in enumerate(set(lt))}
    lt = [l_map[l] for l in lt]

    l_map = {l:i for i, l in enumerate(set(lp))}
    lp = [l_map[l] for l in lp]

    nmi = skmetrics.normalized_mutual_info_score(lt, lp)

    num_k = max((len(set(lp)), len(set(lt))))
    cost = np.zeros((num_k, num_k))
    for i, k in zip(lp, lt):
        cost[i, k] = cost[i, k] + 1
    map_lab = optimize.linear_sum_assignment(-cost)[1]
    lp_map = np.array([map_lab[i] for i in lp])
    acc = np.sum(lp_map==lt) / len(lt)

    purity = []
    lp, lt = np.array(lp), np.array(lt)
    for i in set(lp):
        distri = lt[lp == i]
        purity.append(np.max([np.sum(distri == j) for j in set(distri)])/len(distri))
    purity = np.mean(purity)
    
    return '  NMI %.4f  Acc %2.2f  Pur %2.2f  ' % (nmi, acc*100, purity*100)


def map_lp(lp, lt):
    l_map = {l:i for i, l in enumerate(set(lt))}
    lt = [l_map[l] for l in lt]

    l_map = {l:i for i, l in enumerate(set(lp))}
    lp = [l_map[l] for l in lp]

    num_k = max((len(set(lp)), len(set(lt))))
    cost = np.zeros((num_k, num_k))
    for i, k in zip(lp, lt):
        cost[i, k] = cost[i, k] + 1
    map_lab = optimize.linear_sum_assignment(-cost)[1]
    lp_map = np.array([map_lab[i] for i in lp])
    return lp_map, np.array(lt)

if __name__ == '__main__':
    main()