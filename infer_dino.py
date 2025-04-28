#! /usr/bin/env python3
import os, argparse, numpy as np
import torch.multiprocessing as mp, torch.distributed as dist
from torch.utils.data import DataLoader
from dataset import WavDataset_DINO as WavDataset
from torchaudio import transforms
import torch.nn.parallel, torch.distributed as dist
import torch.utils.data.distributed
from modules.ecapa_tdnn_dino_ali import ECAPA_TDNN
from utils.util import compute_eer
from scipy import spatial
from tqdm import tqdm

class SpeakerModel(torch.nn.Module):
    def __init__(self, feat_dim, embd_dim):
        super(SpeakerModel, self).__init__()
        self.encoder = ECAPA_TDNN(input_size=feat_dim, lin_neurons=embd_dim, channels=[1024, 1024, 1024, 1024, 3072])

    def forward(self, x):
        x = self.encoder(x)
        return x


parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', default='dino_multi_crop_ali', type=str)
parser.add_argument('--model_name', default='model', type=str)
parser.add_argument('--model_num', default='100', type=str)
parser.add_argument('--model_id', default='criterion', type=str)

# validation dataset
parser.add_argument('--val_data_name', default='vox_test', type=str)
parser.add_argument('--val_save_name', default='test', type=str)
parser.add_argument('--scp_name', default='wav', type=str)
parser.add_argument('-j', '--workers', default=4, type=int)
# model backbone
parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--feat_dim', default=80, type=int)
parser.add_argument('--embd_dim', default=512, type=int)
# others
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--scoring', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--onlyscore', default=False, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()

def main(number):

    print(args.onlyscore, args.scoring)
    print(args.val_save_name)
    if not args.onlyscore:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # feature
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

        val_dataset = WavDataset([line.split() for line in open('data/%s/%s.scp' % (args.val_data_name, args.scp_name))], fs=args.fs)
        val_dataloader = DataLoader(val_dataset, num_workers=10, shuffle=False, batch_size=1)

        # dataset
        model = SpeakerModel(args.feat_dim, args.embd_dim)

        ckp = torch.load('exp/%s/%s_%s.pkl' % (args.save_dir, args.model_name, number), map_location='cpu')[args.model_id]
        for i, j in ckp.items():
            if 'module.' in i:
                ckp = {i[7:]:j for i, j in ckp.items()}
                break

        if args.model_id == 'criterion':
            ckp = {i.replace('ema_model.backbone', 'encoder'):j for i, j in ckp.items() if 'ema_model.backbone' in i}
        elif args.model_id == 'ema_teacher':
            ckp = {i.replace('ema_model.encoder', 'encoder'):j for i, j in ckp.items() if 'ema_model.encoder' in i}
        model.load_state_dict(ckp)
        model = model.cuda()
        model.eval()

        embds = {}
        with torch.no_grad():
            for j, (feat, utt) in tqdm(enumerate(val_dataloader)):
                feat = feat.cuda()
                embds[utt[0]] = model(featCal(feat)).cpu().numpy()

        np.save('exp/%s/%s_%s.npy' % (args.save_dir, args.val_save_name,number), embds)
    else:
        embds = np.load('exp/%s/%s_%s.npy' % (args.save_dir, args.val_save_name, number), allow_pickle=True).item()
    # 计算eer
    if args.scoring:
        f = open('exp/%s/%s_scoring.txt' % (args.save_dir, args.val_save_name), 'a' )
        eer,threshold ,cost,_ = get_eer(embds, trial_file='data/%s/trials' % (args.val_data_name))
        f.write('Model:%s  %s_%s.pkl\n' %(args.save_dir, args.model_name, number))
        f.write('EER : %.4f%% Th : %.4f mDCT : %.4f\t\n'%(eer*100, threshold, cost))
        f.flush()

def get_eer(embd_dict1, trial_file):
    true_score = []
    false_score = []

    with open(trial_file) as fh:
        for line in fh:
            line = line.strip()
            key, utt1, utt2,  = line.split()
            result = 1 - spatial.distance.cosine(embd_dict1[utt1][0], embd_dict1[utt2][0])
            if key == '1':
                true_score.append(result)
            elif key == '0':
                false_score.append(result)  
    eer, threshold, mindct, threashold_dct = compute_eer(np.array(true_score), np.array(false_score),p_target=0.05)
    return eer, threshold, mindct, threashold_dct

if __name__ == '__main__':
    number_list = args.model_num.split(',')
    for num in number_list:
        main(num)