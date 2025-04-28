#! /usr/bin/env python3
import os, argparse, numpy as np
import torch.multiprocessing as mp, torch.distributed as dist
from torch.utils.data import DataLoader
from dataset import WavDataset_DINO as WavDataset
from torchaudio import transforms
import torch.nn.parallel, torch.distributed as dist
import torch.utils.data.distributed
from modules.ecapa_tdnn_dino_ali import ECAPA_TDNN_WavLM as ECAPA_TDNN
from utils.util import compute_eer
from scipy import spatial
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from wavlm.WavLM import WavLM, WavLMConfig
from modules.back_classifier import ArcFace

class SpeakerModel(torch.nn.Module):
    def __init__(self, feat_dim, embd_dim):
        super(SpeakerModel, self).__init__()
        self.encoder = ECAPA_TDNN(input_size=feat_dim, lin_neurons=embd_dim, channels=[1024, 1024, 1024, 1024, 3072], log_input=False)
        self.feature_weight = nn.Parameter(torch.zeros(24))
        self.classifier = ArcFace(embd_dim, 8000, s=32.0, m = 0.2)
        

    def forward(self, x):
        norm_weights = F.softmax(self.feature_weight, dim=-1)
        x = torch.tensordot(norm_weights, x, dims=([0], [0]))
        x = x.permute(0, 2, 1) # B,D,T
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
parser.add_argument('--feat_dim', default=1024, type=int)
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

        val_dataset = WavDataset([line.split() for line in open('data/%s/%s.scp' % (args.val_data_name, args.scp_name))], fs=args.fs)
        val_dataloader = DataLoader(val_dataset, num_workers=10, shuffle=False, batch_size=1)

        # dataset
        model = SpeakerModel(args.feat_dim, args.embd_dim)
        model_ptm = WavLM_Large(sub_layers=24)

        ckp = torch.load('exp/dino_ali_lnc_nlm_8k_wavlm_layer24_step2_ft/model_3_499items.pkl', map_location='cpu')[args.model_id]
        ptm = torch.load('exp/dino_ali_lnc_nlm_8k_wavlm_layer24_step2_ft/model_3_499items.pkl', map_location='cpu')['model_ptm']

        for i, j in ckp.items():
            if 'module.' in i:
                ckp = {i[7:]:j for i, j in ckp.items()}
                break
        for i, j in ptm.items():
            if 'module.' in i:
                ptm = {i[7:]:j for i, j in ptm.items()}
                break

        if args.model_id == 'criterion':
            ckp = {i.replace('ema_model.backbone', 'encoder'):j for i, j in ckp.items() if 'ema_model.backbone' in i}
        elif args.model_id == 'ema_teacher':
            ckp = {i.replace('ema_model.', ''):j for i, j in ckp.items() if 'ema_model.' in i}
        model.load_state_dict(ckp)
        model_ptm.load_state_dict(ptm)

        model = model.cuda()
        model_ptm = model_ptm.cuda()
        model.eval()
        model_ptm.eval()

        embds = {}
        with torch.no_grad():
            for j, (feat, utt) in tqdm(enumerate(val_dataloader)):
                if len(feat[0]) > 16000*100:
                    feat = feat[:,:16000*100]
                feat = feat.cuda()
                feat = model_ptm(feat)
                embds[utt[0]] = model(feat).cpu().numpy()

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