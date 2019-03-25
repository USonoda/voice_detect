import pyaudio
import numpy as np
import sys, time
import librosa
import librosa.display


def main():
    CHUNK = 1024  # バッファのサンプル数
    length = 16
    RATE = 16000  # サンプルレート16kHz

    raw_data = np.zeros((CHUNK * length,), dtype='float32')  # 16*1024サンプル →　1秒分

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt8,  # 8bit (int8型)
                    channels=1,  # monaural
                    rate=RATE,
                    frames_per_buffer=CHUNK,
                    input=True)

    while stream.is_active():
        try:
            input_data = stream.read(CHUNK)
            data = np.frombuffer(input_data, dtype='int8')
            data = data.astype('float64') / 128.0

            tmp = raw_data.copy()
            raw_data[:-CHUNK] = tmp[CHUNK:]
            raw_data[-CHUNK:] = data.copy()  # 前（過去）を削ってdata(CHUNK個のサンプル)を後ろに挿入
            # 1秒に16回更新

            melspecs = librosa.feature.melspectrogram(y=raw_data,
                                                      sr=16000,  # Sampling Rate
                                                      n_fft=1024,  # FFT 窓関数のながさ
                                                      n_mels=256,  # Melバンド(縦の画素)の数
                                                      hop_length=128)
            # melspecs のshapeは　(n_mels, t)
            melspecs = librosa.power_to_db(melspecs, ref=np.max)

            melspecs = melspecs[np.newaxis, np.newaxis, :]  # (1,1,n_mels,t)
            print(melspecs.shape)



        except KeyboardInterrupt:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
