
import os, sys
sys.path.append(os.path.dirname(__file__))

# tensorflow warning 제거. tensorflow import 전에 실행해야 함 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from common import logger, args, util, registry
log = logger.make_logger(__name__)


def main():

    o = args.ArgOpt
    opts = [o.TRAIN, o.MODEL, o.BOARD]
    opts_val = args.parse_args(f'BBDR - {util.get_program_name()}', *opts)

    log.info('*'*75)
    log.info(f'tensorflow ver: {tf.__version__}')
    log.info(f'train: {opts_val[o.TRAIN.value]}')
    log.info(f'model: {opts_val[o.MODEL.value]}')
    log.info(f'board: {opts_val[o.BOARD.value]}')
    log.info('*'*75)

    reg = registry.registry
    board = reg[opts_val[o.BOARD.value]](opts_val)
    board.train()

    return


if __name__ == '__main__':
    main()

