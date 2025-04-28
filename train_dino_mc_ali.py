import os, time, random, argparse, numpy as np, sys
import torch, torch.nn as nn, torch.nn.functional as F
import librosa, scipy.io.wavfile as sciwav, torchaudio
from scipy.signal import fftconvolve
import torch.multiprocessing as mp, torch.distributed as dist, torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
from modules.ecapa_tdnn_dino_ali import ECAPA_TDNN, RDINOHead
from dataset import WavBatchDistributedSampler, WavDataset_DINO as WavDataset
from utils.utils import AverageMeter, ProgressMeter, save_ramdom_state, get_lr
from utils.spk_veri_metric import  SVevaluation
from nemo.core.optim.lr_scheduler import CosineAnnealing
from torchaudio import transforms

# nohup python train_dino_mc_ali.py > dino_mc_ali_clodstart.log &


parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', default='dino_multi_crop_ali_cold_start', type=str)
# model setups
parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--feat_dim', default=80, type=int)
parser.add_argument('--embd_dim', default=512, type=int)
parser.add_argument('--num_clusters', default=65536, type=int)
parser.add_argument('--ema_decay', default=[0.999, 0.9999], nargs='+', type=float)
parser.add_argument('--clip_grad', default=3.0, type=float)
parser.add_argument('--freeze_prototypes_epoch', default=1, type=int)
# data augmentation
parser.add_argument('--crop_dur', default=[2, 4], nargs='+', type=float)
parser.add_argument('--crop_num', default=[4, 2], nargs='+', type=int)
parser.add_argument('--snr_range', default=[0, 20], nargs='+', type=int)
# training dataset
parser.add_argument('--data_name', default='vox2dev', type=str)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=240, type=int)
# test dataset
parser.add_argument('--val_data_name', default='vox_test', type=str)
parser.add_argument('--val_freq', default=1, type=int)
# eer and cost
parser.add_argument('--ptar', default=[0.01], nargs='+', type=float)
# optimizer
parser.add_argument('--wd', '--weight_decay', default=5e-5, type=float)
# learning rate scheduler
parser.add_argument('--lr', default=0.25, type=float)
parser.add_argument('--min_lr', default=1e-5, type=float)
parser.add_argument('--warm_up_epoch', default=10, type=int)
# others
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
# single node distributed parallel training
parser.add_argument('--seed', default=3007123, type=int)
parser.add_argument('-p', '--print-freq', default=10, type=int)
parser.add_argument('--port', default='8229', type=str)
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)

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
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%s' % args.port, world_size=args.ngpus, rank=args.rank)
        args.batch_size = int(args.batch_size / args.ngpus)
        args.workers = int((args.workers + args.ngpus - 1) / args.ngpus)

    utt2wav = [line.split() for line in open('data/%s/wav.scp' % args.data_name)]
    spk2int = {line.split()[0]:i for i, line in enumerate(open('data/%s/spk2utt' % args.data_name))}
    utt2spk = {line.split()[0]:spk2int[line.split()[1]] for line in open('data/%s/utt2spk' % args.data_name)}
    noise_list = {'noise': [i.strip('\n') for i in open('data/envir/noise_wav_list')],
                  'music': [i.strip('\n') for i in open('data/envir/music_wav_list')],
                  'babb': [i.strip('\n') for i in open('data/envir/speech_wav_list')],
                  'reverb': [i.strip('\n') for i in open('data/envir/simu_rir_list')]}

    trn_dataset = WavDataset(utt2wav, utt2spk, args.fs, is_aug=True, snr=args.snr_range, noise_list=noise_list, crop_dur=args.crop_dur, crop_num=args.crop_num)
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

    # spectral feature calculation
    featCal = transforms.MelSpectrogram(
                sample_rate=args.fs,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=0.0,
                f_max=8000,
                pad=0,
                n_mels=args.feat_dim
            ).cuda()
    featCal.eval()

    # create model
    model = SpeakerModel(args.feat_dim, args.embd_dim, num_spks=args.num_clusters)
    #model.load_state_dict(torch.load(f'exp/{args.save_dir}/dino_pretrained.pth'))

    criterion = SpeakerModel(args.feat_dim, args.embd_dim, num_spks=args.num_clusters)
    criterion.load_state_dict(model.state_dict())
    criterion = EMAteacher(criterion, num_clusters=args.num_clusters, ema_decay=args.ema_decay,
                           ema_anneal_end_step=len(trn_loader)*args.epochs)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        criterion = torch.nn.SyncBatchNorm.convert_sync_batchnorm(criterion)
        criterion.cuda(args.gpu)
    else:
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
    
    # optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(get_params_groups(model, args.wd), lr=args.lr, momentum=0.9)
    scheduler = CosineAnnealing(optimizer, warmup_steps=len(trn_loader)*args.warm_up_epoch, max_steps=len(trn_loader)*args.epochs, min_lr=args.min_lr)

    # optionally resume from a checkpoint
    if args.start_epoch != 0:
        checkpoint = torch.load('exp/%s/model_%d.pkl' % (args.save_dir, args.start_epoch-1), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        criterion.load_state_dict(checkpoint['criterion'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        checkpoint = torch.load('exp/%s/random_state.pkl' % (args.save_dir), map_location='cpu')
        random.setstate(checkpoint['random'])
        np.random.set_state(checkpoint['np'])
        torch.set_rng_state(checkpoint['torch'])
    elif args.gpu == 0:
        print(str(model), flush=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            trn_sampler.set_epoch(epoch)

        trn_loss = train(trn_loader, featCal, model, criterion, optimizer, scheduler, epoch, args)

        if epoch % args.val_freq == 0:
            embd = test(tst_loader, criterion.ema_model, featCal, args)[:len(val_utt)]
            if args.rank == 0:
                eer_cal.update_embd(embd)
                eer, cost = eer_cal.eer_cost()
                print(f'Validate  Epoch {epoch:3d}\tTrainLoss {trn_loss:2.4f}\tLR {get_lr(optimizer)}\tEER {eer:.4f}\tcost {cost:.4f}\n', flush=True)
        elif args.rank == 0:
                print(f'Validate  Epoch {epoch:3d}\tTrainLoss {trn_loss:2.4f}\tLR {get_lr(optimizer)}\n', flush=True)


def train(trn_loader, featCal, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    var_ts = AverageMeter('VarT', ':6.2f')
    var_ps = AverageMeter('VarP', ':6.2f')
    progress = ProgressMeter(len(trn_loader), [batch_time, data_time, losses, var_ts, var_ps], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (local_view, global_view, _) in enumerate(trn_loader):
        data_time.update(time.time() - end)

        bs, num_crop, length = local_view.shape
        local_view = local_view.reshape(bs*num_crop, length)
        local_view = featCal(local_view.cuda(args.gpu, non_blocking=True))
        local_view = model(local_view).reshape(bs, num_crop, -1)

        bs, num_crop, length = global_view.shape
        global_view = global_view.reshape(bs*num_crop, length)
        global_view = featCal(global_view.cuda(args.gpu, non_blocking=True))
        global_view_student = model(global_view).reshape(bs, num_crop, -1)

        loss, target_var, pred_var = criterion(local_view, global_view_student, global_view, model)

        losses.update(loss.item(), bs)
        var_ts.update(target_var.item(), bs)
        var_ps.update(pred_var.item(), bs)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model, args.clip_grad)
        cancel_prototypes_gradients(epoch, model, args.freeze_prototypes_epoch)
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, suffix='LR %.6f'  % get_lr(optimizer))

        if args.rank == 0 and i == len(trn_loader)-1:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'criterion': criterion.state_dict()},
                        f'exp/{args.save_dir}/model_{epoch}.pkl')
            save_ramdom_state('exp/%s' % args.save_dir, random.getstate(), np.random.get_state(), torch.get_rng_state())

    if args.distributed:
        loss = torch.tensor(losses.avg).cuda(args.gpu)
        dist.all_reduce(loss, op=dist.reduce_op.SUM)
        return loss / args.ngpus
    else:
        return losses.avg


def test(tst_loader, model, featCal, args):
    model.eval()
    
    with torch.no_grad():
        for j, (feats, _) in enumerate(tst_loader):
            feats = feats.cuda(args.gpu, non_blocking=True)
            feats = featCal(feats)
            embd = model(feats, encoder_output=True)
            all_embd = embd if j == 0 else torch.cat((all_embd, embd))
                
    if args.distributed:
        gather_embd = [torch.zeros_like(all_embd).cuda(args.gpu) for i in range(dist.get_world_size())]
        dist.all_gather(gather_embd, all_embd)
        all_embd = torch.cat(gather_embd)

    return all_embd.cpu().numpy()


class EMAteacher(nn.Module):
    def __init__(self, model, num_clusters,
                 ema_decay=[0.996, 0.9999], ema_anneal_end_step=None,
                 min_target_var=0.1, min_pred_var=0.01):
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
        
        self.min_target_var = min_target_var
        self.min_pred_var = min_pred_var

        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))
        self.register_buffer("centers", torch.zeros((1, num_clusters), dtype=torch.float32))

        self.center_ema_decay = 0.9

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

    def forward(self, local_view, global_view_student, global_view, new_model):
        if self.training:
            self.set_decay()
            self.ema_step(new_model)
        
        with torch.no_grad():
            self.ema_model.eval()
            global_view = self.ema_model(global_view)
            ema_teacher_output = global_view

            target_var = self.compute_var(global_view)
            pred_var = self.compute_var(local_view.float())

        global_view = F.softmax((global_view - self.centers) / 0.04, dim=-1).reshape(global_view_student.shape)
        local_view = F.log_softmax(local_view / 0.1, dim=-1)
        global_view_student = F.log_softmax(global_view_student / 0.1, dim=-1)

        total_loss = 0
        n_loss_terms = 0
        for i in range(global_view.shape[1]):
            for j in range(global_view_student.shape[1]):
                if i == j:
                    continue
                total_loss -= (global_view_student[:, j] * global_view[:, i]).sum(dim=-1).mean()
                n_loss_terms += 1
            for j in range(local_view.shape[1]):
                total_loss -= (local_view[:, j] * global_view[:, i]).sum(dim=-1).mean()
                n_loss_terms += 1
        
        self.update_center(ema_teacher_output)
        return total_loss / n_loss_terms, target_var, pred_var
    
    @torch.no_grad()
    def update_center(self, ema_teacher_output):
        if dist.is_initialized():
            batch_center = torch.sum(ema_teacher_output, dim=0, keepdim=True)
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(ema_teacher_output) * dist.get_world_size())
        else:
            batch_center = ema_teacher_output.mean(dim=0)
        self.centers = self.centers * self.center_ema_decay + batch_center * (1 - self.center_ema_decay)

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()


class SpeakerModel(torch.nn.Module):
    def __init__(self, feat_dim, embd_dim, num_spks=6000):
        super(SpeakerModel, self).__init__()
        self.backbone = ECAPA_TDNN(input_size=feat_dim, lin_neurons=embd_dim, channels=[1024, 1024, 1024, 1024, 3072])
        self.head = RDINOHead(in_dim=embd_dim, out_dim=num_spks, use_bn=True)

    def forward(self, x, encoder_output=False):
        x = self.backbone(x)
        if encoder_output:
            return x
        output = self.head(x)
        return output[1]


def get_params_groups(model, weight_decay=5e-5):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized, 'weight_decay': weight_decay},
                {'params': not_regularized, 'weight_decay': 0.}]

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def cancel_prototypes_gradients(epoch, model, freeze_prototypes_epoch):
    if epoch >= freeze_prototypes_epoch:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


if __name__ == '__main__':
    main()