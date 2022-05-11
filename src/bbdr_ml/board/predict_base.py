
import os
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from common import args, util, const


class PredictBase:

    def __init__(self, opts_val) -> None:
        self.opts_val = opts_val
    
        model, vocab = self._load_model_vocab()
        self.model = model
        self.vocab = vocab

        index_word = util.get_index_to_word(vocab)
        self.index_word = index_word


    def predict(self):
        o = args.ArgOpt
        predict_csv = self.opts_val[o.PREDICT.value]

        df = pd.read_csv(predict_csv, encoding='utf-8-sig')
        df = df.drop_duplicates(subset = ['uid'])
        print(df)

        df['useful']  = df['content'].map(self._spam_predict)
        df_out = df[['content', 'useful']]
        print(df_out)

        _, tmp_file = os.path.split(predict_csv)
        out_file = f'predict_{tmp_file}'
        print(f'tmp_file: {tmp_file}, out_file: {out_file}')

        if len(df_out) > 0:
            df_out.to_csv(out_file, index=False, header=False, encoding='utf-8-sig')


    def _load_model_vocab(self):
        o = args.ArgOpt
        model_base = self.opts_val[o.MODEL.value]
        model_dir = f'{util.get_program_path()}/model'
        model_path = f'{model_dir}/{model_base}.model'
        vocab_path = f'{model_dir}/{model_base}.vocab'

        model = load_model(model_path)
        vocab = util.load_dictionary(vocab_path)

        return model, vocab


    def _spam_predict(self, content):

        model = self.model
        vocab = self.vocab
        index_word = self.index_word

        # new_sentence = content
        # new_sentence = text_preprocess.only_hangul(content)
        # new_sentence = text_preprocess.morpheme_komoran(new_sentence)
        #print(new_sentence)

        # 정수 인코딩
        encoded = []
        try:
            for word in content.split():
                try:
                    encoded.append(vocab[word])
                except:
                    encoded.append(0)
        except:
            pass
        
        #print(encoded)
        # index_to_word = util.get_index_to_word(vocab)
        print()
        print(' '.join([index_word[index] for index in encoded]))
        #print(token.sequences_to_texts(encoded))
        pad_new = pad_sequences([encoded], maxlen=const.Const.DEFAULT_PAD_LENGTH)
        score = float(model.predict(pad_new))
        #print()

        if(score > 0.5):
            print(f'[ OOOOO ] {int(score * 100)}% - 유용')
            ret = True
        else:
            print(f'[ XXXXX ] {int(score * 100)}% - 불필요')
            ret = False

        return ret