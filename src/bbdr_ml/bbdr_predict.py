
import os, sys
sys.path.append(os.path.dirname(__file__))

# tensorflow warning 제거. tensorflow import 전에 실행해야 함 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from common import logger, args, util
from board.predict_base import PredictBase
log = logger.make_logger(__name__)


def main():

    print(f'__file__: {__file__}')
    o = args.ArgOpt
    opts = [o.PREDICT, o.MODEL]
    opts_val = args.parse_args(f'BBDR - {util.get_program_name()}', __file__, *opts)

    log.info('*'*75)
    log.info(f'tensorflow ver: {tf.__version__}')
    log.info(f'predict: {opts_val[o.PREDICT.value]}')
    log.info(f'model: {opts_val[o.MODEL.value]}')
    log.info('*'*75)

    pre = PredictBase(opts_val)
    pre.predict()

    return


if __name__ == '__main__':
    main()

