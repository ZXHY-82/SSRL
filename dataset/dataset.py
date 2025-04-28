import os, math, random, warnings, torch, librosa, numpy as np, scipy.io.wavfile as sciwav, torchaudio
from scipy.signal import fftconvolve
from python_speech_features import sigproc
from torch.utils.data import Dataset

class WavDataset_DINO(Dataset):
    def __init__(self, wav_scp, utt2label=None, fs=16000,
                is_aug=False, snr=None, noise_list=None, 
                crop_dur=None, crop_num=None):
        self.wav_scp = wav_scp
        self.utt2label = utt2label
        self.fs = fs

        self.is_aug=is_aug
        self.snr = snr
        self.noise_list = noise_list

        self.crop_dur = crop_dur
        self.crop_num = crop_num
        self.preemph = 0.97

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
        noise_types = random.choice(['reverb', 'noise']*2+['none']) # 80%
        
        if  'reverb' in noise_types:
            power = (signal ** 2).mean()
            rir = self._load_data(random.choice(self.noise_list['reverb']))
            rir = (rir - min(rir)) / (max(rir) - min(rir))
            max_ind = np.argmax(np.abs(rir))
            rir = rir[max_ind:]
            signal = fftconvolve(signal, rir)[:signal.shape[0]]
            power2 = (signal ** 2).mean()
            signal = np.sqrt(power / max(power2, 1e-10)) * signal

        if  'noise' in noise_types:
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
            signal = signal + noise_signal * sigma_n
        
        return signal

    def _truncate_speech(self, signal, tlen, offset=None):
        if tlen == None:
            return signal
        if signal.shape[0] <= tlen:
            signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
        if offset is None:
            offset = random.randint(0, signal.shape[0] - tlen)
        return np.array(signal[offset : offset + tlen])

    def __getitem__(self, idx):
        utt, wav = self.wav_scp[idx]
        label = self.utt2label[utt] if self.utt2label else utt
        wav = self._load_data(wav)

        if not self.is_aug:
            wav = self._norm_speech(wav)
            wav = sigproc.preemphasis(wav, self.preemph)
            wav = torch.from_numpy(wav.astype('float32'))
            return wav, label

        short_wavs = []
        dur = int(self.crop_dur[0] * self.fs)
        for i in range(self.crop_num[0]):
            one_wav = self._truncate_speech(wav, dur)
            one_wav = self._augmentation(one_wav)
            one_wav = self._norm_speech(one_wav)
            one_wav = sigproc.preemphasis(one_wav, self.preemph)
            one_wav = torch.from_numpy(one_wav.astype('float32'))
            short_wavs.append(one_wav)
        short_wavs = torch.stack(short_wavs)

        long_wavs = []
        dur = int(self.crop_dur[1] * self.fs)
        for i in range(self.crop_num[1]):
            one_wav = self._truncate_speech(wav, dur)
            one_wav = self._augmentation(one_wav) # DINO自监督训练，short跟long wav都加噪
            one_wav = self._norm_speech(one_wav)
            one_wav = sigproc.preemphasis(one_wav, self.preemph)
            one_wav = torch.from_numpy(one_wav.astype('float32'))
            long_wavs.append(one_wav)
        long_wavs = torch.stack(long_wavs)
        
        return short_wavs, long_wavs, label