
import argparse
import re
from enum import Enum

from common import logger, const
log = logger.make_logger(__name__)


class ArgOpt(Enum):
    DOMAIN = 'domain'
    CATEGORY = 'category'
    SITE = 'site'
    KEYWORD = 'keyword'
    START = 'start'
    END = 'end'


def parse_args(desc, *args):
    parser = argparse.ArgumentParser(description=desc)
    o = ArgOpt

    # setup arg options
    for arg in args:
        arg: ArgOpt
        if arg == o.DOMAIN:
            domains = list(const.Const.DOMAIN.keys())
            help_domain = ','.join(domains)
            parser.add_argument('-d', f'--{arg.value}', help=help_domain)

        elif arg == o.START:
            parser.add_argument('-s', f'--{arg.value}')

        elif arg == o.END:
            parser.add_argument('-e', f'--{arg.value}')

    # parse arg options
    args_dict = vars(parser.parse_args())
    args_keys = args_dict.keys()

    ret = {}
    for key, value in args_dict.items():
        if key == o.DOMAIN.value:
            ret[key] = _split_multi_args(value, domains)

        elif key == o.START.value or key == o.END.value:
            _check_date_format(key, value)
            ret[key] = value

        else:
            log.warn(f'unknown args: {key} ({value})')
            ret[key] = value

    # start and end
    if o.START.value in args_keys and o.END.value in args_keys:
        start = args_dict[o.START.value]
        end = args_dict[o.END.value]
        if start > end:
            print(f'[ERROR] Start must have an earlier date than end. start: {start}, end: {end}')
            exit(1)
        
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

