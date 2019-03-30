import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.callbacks import Callback, CSVLogger
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

input_r, input_c = (256,126)


class PlotLosses(Callback):
    # 学習中のlossについてlive plotする

    def on_train_begin(self, logs={}):
        '''
        訓練開始時に実施
        '''
        self.epoch_cnt = 0      # epochの回数を初期化
        plt.axis([0, self.epochs, 0, 0.25])
        plt.ion()               # pyplotをinteractive modeにする

    def on_train_end(self, logs={}):
        '''
        訓練修了時に実施
        '''
        plt.ioff()              # pyplotのinteractive modeをoffにする
        plt.legend(['loss', 'val_loss'], loc='best')
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
        '''
        epochごとに実行する処理
        '''
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        x = self.epoch_cnt
        # epochごとのlossとval_lossをplotする
        plt.scatter(x, loss, c='b', label='loss')
        plt.scatter(x, val_loss, c='r', label='val_loss')
        plt.pause(0.05)
        # epoch回数をcount up
        self.epoch_cnt += 1


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


def main(epochs=50, batch_size=128):
    x = np.load('./dataset/dataset_x.npy')
    y = np.load('./dataset/dataset_y.npy')

    m = np.max(x)
    print(m)
    x = x.reshape(x.shape[0], input_r, input_c, 1).astype('float32') / m  # reshapeうまくいくのか？
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert one-hot vector
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(input_r,input_c,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    print(model.summary())

    # callback function
    plot_losses = PlotLosses()      # グラフ表示(live plot)
    plot_losses.epochs = epochs
    csv_logger = CSVLogger('trainlog.csv')

    # train
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[plot_losses, csv_logger])

    # result
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {0}'.format(score[0]))
    print('Test accuracy: {0}'.format(score[1]))
    model.save('voice.h5')
    plot_result(history)


if __name__ == '__main__':
    main()
