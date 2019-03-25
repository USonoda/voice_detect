from __future__ import print_function
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os,time

seiyu = ['ayaneru']
sampling_rate = 16000

def mel(ndarray):
    melspecs = librosa.feature.melspectrogram(y=ndarray,
                                              sr=sampling_rate,  # Sampling Rate
                                              n_fft=1024,  # FFT window のながさ
                                              n_mels=256,  # Melバンドの数(縦の画素数)
                                              hop_length=128)  # 横軸の移動幅　
    melspecs = librosa.power_to_db(melspecs, ref=np.max)  # np.array (256,173)
    melspecs = melspecs.astype(np.float32)
    return melspecs


def draw(m):
    librosa.display.specshow(m, fmax=sampling_rate, cmap='coolwarm')
    plt.tight_layout()
    plt.show()


def main():
    a = time.time()
    x = []
    y = np.array([])
    for n in range(len(seiyu)):
        s = seiyu[n]
        dir = './dataset/{}/'.format(s)
        files = os.listdir(dir)
        c = 0
        for f in files:
            if f[0] == '.':
                continue
            y, sr = librosa.load(dir+f,sr=sampling_rate)
            dur_time = len(y)//sampling_rate
            print(f, dur_time)

            for i in range(dur_time):
                melspec = mel(y[sampling_rate*i: sampling_rate*(i+1)])
                x.append(melspec)
                c += 1

        np.append(y, [n for _ in range(c)])
    x = np.array(x)
    np.save('./dataset/dataset.npy',np.array([x,y]))
    print(time.time() - a)


if __name__ == '__main__':
    main()
