import torch, torch.nn as nn, random
from torchaudio import transforms

class logFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(logFbankCal, self).__init__()
        self.fbankCal = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  n_mels=n_mels)

    def forward(self, x, is_aug=[]):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
                
        return out