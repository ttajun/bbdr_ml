
# import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

from common import logger, args, util
log = logger.make_logger(__name__)


class BoardBase:

    def __init__(self, opts_val) -> None:
        self.opts_val = opts_val


    def train(self):
        o = args.ArgOpt

        log.info(f'Step 1. 데이터 로드')
        df = self._1_load_data(self.opts_val[o.TRAIN.value])
        log.info(df)

        log.info(f'Step 2. 전처리')
        df = self._2_preprocess(df)
        log.info(f'{df.groupby("useful").size().reset_index(name="count")}')

        log.info(f'Step 3. 토큰화 및 정수인코딩')
        X_data = df['content']
        Y_data = df['useful']

        # content를 vocab에 있는 index 집합으로 변경
        # '나는 ..' => '234 123 ..'
        X_data, vocab = self._3_tokenize(X_data)
        vocab_size = len(vocab) + 1

        log.info(f'Step 4. 모델 생성')
        model = self._4_fit_model(X_data, Y_data, vocab_size)

        self._save_model_vocab(model, vocab)


    def _save_model_vocab(self, model, vocab):
        o = args.ArgOpt
        model_dir = f'{util.get_program_path()}/model'
        util.create_directory(model_dir)

        model_base = self.opts_val[o.MODEL.value]
        model_path = f'{model_dir}/{model_base}.model'
        vocab_path = f'{model_dir}/{model_base}.vocab'
        model.save(model_path)
        util.save_dictionary(vocab, vocab_path)


    def _1_load_data(self, train):
        return pd.read_csv(train, encoding='utf-8-sig')


    def _2_preprocess(self, df:DataFrame):
        # useful 필드에는 문자'True'가 있으나 df로 받아오면서 bool로 변경됨
        # useful 필드의 'True'(bool)를 1(숫자)로 변경한다.
        df['useful'] = df['useful'].replace([True, False], [1, 0])

        # content의 한글 제외 삭제
        df['content'] = df['content'].map(util.only_hangul)

        # Null 제거
        df = df.dropna(axis=0)

        # 중복 제거
        # df = df.drop_duplicates(subset = ['content'])

        return df


    def _3_tokenize(self, X_data):
        token = Tokenizer()
        token.fit_on_texts(X_data)
        sequences = token.texts_to_sequences(X_data)
        # print(sequences[:5])

        word_to_index = token.word_index
        # print(f'{repr(word_to_index)}')

        # 단어 출현 빈도수
        threshold = 2                   # 빈도수 제한
        total_cnt = len(word_to_index)  # 단어 수
        rare_cnt = 0                    # 등장 빈도수가 threshold보다 작은 단어의 개수
        total_freq = 0                  # 훈련 데이터의 전체 단어 빈도수 총 합
        rare_freq = 0                   # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수 총 합

        for key, value in token.word_counts.items():
            total_freq = total_freq + value

            if value < threshold:
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        print(f'등장 빈도가 {threshold - 1}번 이하인 희귀 단어의 수: {rare_cnt}')
        print(f'단어 집합(vocabulary)에서 희귀 단어의 비율: {(rare_cnt / total_cnt)*100}')
        print(f'전체 등장 빈도에서 희귀 단어 등장 빈도 비율: {(rare_freq / total_freq)*100}')

        vocab_size = len(word_to_index) + 1
        print(f'단어 집합의 크기: {vocab_size}')

        return sequences, word_to_index


    def _4_fit_model(self, X_data, Y_data, vocab_size, train_ratio=0.8, pad_len=1000):
        n_of_train = int(len(X_data) * train_ratio)
        n_of_test = int(len(X_data) - n_of_train)
        print(f'훈련 데이터 개수: {n_of_train}')
        print(f'테스트 데이터 개수: {n_of_test}')
        print(f'본문의 최대 길이: {max(len(l) for l in X_data)}')
        print(f'본문의 평균 길이: {(sum(map(len, X_data))/len(X_data))}')

        data = pad_sequences(X_data, maxlen=pad_len)

        x_test = data[n_of_train:]
        y_test = np.array(Y_data[n_of_train:])
        x_train = data[:n_of_train]
        y_train = np.array(Y_data[:n_of_train])

        model = Sequential()
        model.add(Embedding(vocab_size, 32))
        model.add(SimpleRNN(32))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(x_train, y_train, epochs=4, batch_size=64, validation_split=0.2)
        print(f'\n 테스트 정확도: {(model.evaluate(x_test, y_test)[1])}')

        epochs = range(1, len(history.history['acc']) + 1)
        plt.plot(epochs, history.history['loss'])
        plt.plot(epochs, history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        return model


