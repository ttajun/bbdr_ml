
import pandas as pd
from pandas.core.frame import DataFrame

from common import logger, args, util
from board.base import BoardBase
log = logger.make_logger(__name__)


class BoardBasic(BoardBase):

    def __init__(self, opts_val) -> None:
        super().__init__(opts_val)


    # def train(self):
    #     o = args.ArgOpt
    #     train = self.opts_val[o.TRAIN.value]

    #     ## Step 1. 데이터 로드
    #     log.info(f'Step 1. 데이터 로드')
    #     df: DataFrame
    #     df = pd.read_csv(train, encoding='utf-8-sig')
    #     log.info(df)

    #     ## Step 2. 전처리
    #     log.info(f'Step 2. 전처리')

    #     # useful 필드에는 문자'True'가 있으나 df로 받아오면서 bool로 변경됨
    #     # useful 필드의 'True'(bool)를 1(숫자)로 변경한다.
    #     df['useful'] = df['useful'].replace([True, False], [1, 0])

    #     # content의 한글 제외 삭제
    #     df['content'] = df['content'].map(util.only_hangul)

    #     # Null 제거
    #     df = df.dropna(axis=0)

    #     # 중복 제거
    #     # df = df.drop_duplicates(subset = ['content'])

    #     print()
    #     print('유용(1) / 불필요(0)')
    #     print(f'{df.groupby("useful").size().reset_index(name="count")}')
    #     print()

    #     print(df)