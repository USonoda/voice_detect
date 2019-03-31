import keras
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


input_r, input_c = (256,126)
seiyu = ['kokoa','chino','rize','chiya','syaro','bgm']
class_size = len(seiyu)
batch_size = 128
epochs = 200
m = -80


# class PlotLosses(Callback):
#     # 学習中のlossについてlive plotする
#
#     def on_train_begin(self, logs={}):
#         '''
#         訓練開始時に実施
#         '''
#         self.epoch_cnt = 0      # epochの回数を初期化
#         plt.axis([0, self.epochs, 0, 0.25])
#         plt.ion()               # pyplotをinteractive modeにする
#
#     def on_train_end(self, logs={}):
#         '''
#         訓練修了時に実施
#         '''
#         plt.ioff()              # pyplotのinteractive modeをoffにする
#         plt.legend(['loss', 'val_loss'], loc='best')
#         plt.show()
#
#     def on_epoch_end(self, epoch, logs={}):
#         '''
#         epochごとに実行する処理
#         '''
#         loss = logs.get('loss')
#         val_loss = logs.get('val_loss')
#         x = self.epoch_cnt
#         # epochごとのlossとval_lossをplotする
#         plt.scatter(x, loss, c='b', label='loss')
#         plt.scatter(x, val_loss, c='r', label='val_loss')
#         plt.pause(0.05)
#         # epoch回数をcount up
#         self.epoch_cnt += 1


def plot_result(history):
    #plotとsave

    # accuracy
    plt.figure()
    plt.plot(history.history['acc'], label='acc', marker='.')
    plt.plot(history.history['val_acc'], label='val_acc', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('accuracy')
    plt.savefig('graph_accuracy.png')
    plt.show()

    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='loss', marker='.')
    plt.plot(history.history['val_loss'], label='val_loss', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('loss')
    plt.savefig('graph_loss.png')
    plt.show()


def main():
    x = np.load('./dataset/dataset_x.npy')
    y = np.load('./dataset/dataset_y.npy')

    x = x.reshape(x.shape[0], input_r, input_c, 1).astype('float32') / m
    x = np.concatenate([x,x,x], axis=3)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')

    # convert one-hot vector
    y_train = keras.utils.to_categorical(y_train, class_size)
    y_val = keras.utils.to_categorical(y_val, class_size)

    # create model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(input_r,input_c,3))

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten())
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(class_size, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    # train
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val))

    # result
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test loss: {0}'.format(score[0]))
    print('Test accuracy: {0}'.format(score[1]))
    model.save('voice.h5')
    plot_result(history)


if __name__ == '__main__':
    main()
