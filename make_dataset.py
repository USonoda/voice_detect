from __future__ import print_function
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os,time

seiyu = ['kokoa','chino','rize','chiya','syaro','bgm']
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
        dir = './dataset/{}/'.format(s)  # ディレクトリ名をseiyuリストと合わせておく
        files = os.listdir(dir)
        c = 0
        for f in files:
            if f[0] == '.':
                continue
            S, sr = librosa.load(dir+f,sr=sampling_rate)
            dur_time = len(S)//sampling_rate
            print(f, dur_time)

            for i in range(dur_time*4):
                melspec = mel(S[sampling_rate//4*i: sampling_rate//4*(i+4)])
                if melspec.shape != (256,126):
                    continue
                x.append(melspec)
                c += 1
        np.append(y, [n for _ in range(c)])
        print(y.shape)
    x = np.array(x)
    np.save('./dataset/dataset_x.npy', x)
    np.save('./dataset/dataset_y.npy', y)
    print(time.time() - a)


if __name__ == '__main__':
    main()
