
import os
import argparse
import re
from enum import Enum

from common import logger, const, util
log = logger.make_logger(__name__)


class ArgOpt(Enum):
    TRAIN = 'train'
    MODEL = 'model'
    BOARD = 'board'
    PREDICT = 'predict'
    DOMAIN = 'domain'
    START = 'start'
    END = 'end'


def parse_args(desc, file, *args):
    parser = argparse.ArgumentParser(description=desc)
    o = ArgOpt
    con = const.Const
    print(f'file: {file}')

    # setup arg options
    for arg in args:
        arg: ArgOpt
        if arg == o.TRAIN:
            parser.add_argument('-t', f'--{arg.value}')
        elif arg == o.MODEL:
            parser.add_argument('-m', f'--{arg.value}')
        elif arg == o.BOARD:
            help_board = ','.join(con.BOARDS)
            parser.add_argument('-b', f'--{arg.value}', help=help_board)
        elif arg == o.PREDICT:
            parser.add_argument('-p', f'--{arg.value}')

    # parse arg options
    args_dict = vars(parser.parse_args())

    # validation
    ret = {}
    for key, value in args_dict.items():
        # print(f'key: {key}, value: {value}')
        if 'mmaker' in file:
            if key == o.TRAIN.value:
                # 훈련데이터 파일확인
                if not value or not os.path.isfile(value):
                    print(f'train csv is not exist. ({value})')
                    exit(1)
                ret[key] = value
        
            elif key == o.BOARD.value:
                if not value or value not in help_board:
                    print(f'select board in [{help_board}]. use -b option.')
                    exit(1)
                ret[key] = value

            elif key == o.MODEL.value:
                if not value:
                    print(f'model name must be set. use -m option.')
                    exit(1)
                ret[key] = value

        else:
            if key == o.PREDICT.value:
                # 예측데이터 파일확인
                if not value or not os.path.isfile(value):
                    print(f'predict csv is not exist. ({value})')
                    exit(1)
                ret[key] = value

            elif key == o.MODEL.value:
                # 모델확인
                model_path = f'{util.get_program_path()}/model/{value}.vocab'

                if not value or not os.path.isfile(model_path):
                    print(f'model is not exist. ({model_path})')
                    exit(1)
                ret[key] = value

    return ret


def _check_date_format(key, value):
    pattern = re.compile('((202[0-9]|201[0-9]|200[0-9]|[0-1][0-9]{3})(1[0-2]|0[1-9])(3[01]|[0-2][1-9]|[12]0))')
    match = pattern.match(value)
    if not match:
        print(f'[ERROR] ({value}). {key} format must be YYYYMMDD. ex) 20210601')
        exit(1)

    if len(value) != 8:
        print(f'[ERROR] ({value}). {key} length must be 8.')
        exit(1)


def _split_multi_args(arg, arg_list):
    try:
        tmp = arg.split(',')
        tmp = [i.strip() for i in tmp]
        ret = [i for i in tmp if i in arg_list]
        return ret
    except:
        log.warn(f'_split_multi_args() fail. arg: {arg}, arg_list: {arg_list}')
        return []

