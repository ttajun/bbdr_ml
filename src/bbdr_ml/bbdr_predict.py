
import os, sys
sys.path.append(os.path.dirname(__file__))
import signal

from time import sleep, time
from datetime import datetime as dt, timedelta

from common import args, logger, order, util, const 
from common import result, selenium, queue, registry
log = logger.make_logger(__name__)


def main():

    ##############################
    ### signal handler
    ##############################
    def cleanup():
        res.save_row_part(rows)
        ord._save_order()

    def term_handler(signum, frame):
        log.info(f'>>> INTERRUPT SIGINT !!!. signum: {repr(signum)}')
        cleanup()
        # sel.get_driver().quit()
        exit(1)

    signal.signal(signal.SIGINT, term_handler)

    ##############################
    ### argparse
    ##############################
    o = args.ArgOpt
    opts = [o.DOMAIN, o.START, o.END]
    opts_val = args.parse_args(f'BBDR - {util.get_program_name}', *opts)

    # variables
    start = opts_val[o.START.value]
    end = opts_val[o.END.value]
    execute_domains = opts_val[o.DOMAIN.value]

    start_p = dt.strptime(start, '%Y%m%d')
    end_p = dt.strptime(end, '%Y%m%d')

    ##############################
    ### class
    ##############################
    ord = order.Order(opts_val)
    lev = order.LoopLevel
    con = const.Const
    res = result.Result(ord)

    sel = selenium.Selenium()
    driver = sel.get_driver()

    que = queue.TaskQueue()
    reg = registry.registry

    ##############################
    ### main loop
    ##############################
    rows = []
    for domain, domain_info in con.DOMAIN.items():

        # 수집종료 후 같은 order가 들어온 경우 종료
        if ord.meta_complete_get(): break

        # 선택 도메인만 수집
        if domain not in execute_domains: continue

        # 수집된 도메인 생략 (order)
        if not ord.compare_order_by_list(lev.FIRST, domain, list(con.DOMAIN.keys())): continue
        # print(f'> domain: {domain}')
        # sleep(1)

        # Board
        board = reg[con.SITE_NAVER_NEWS]()

        tmp_p = start_p
        while tmp_p <= end_p:

            str_date = util.date_to_str(tmp_p)
            if not ord.compare_order_by_value(lev.SECOND, str_date): 
                tmp_p += timedelta(days=1)
                continue
            # print(f'> domain: {domain}, date: {str_date}')
            # sleep(1)

            for keyword, search_words in domain_info.items():

                if not ord.compare_order_by_list(lev.THIRD, keyword, list(domain_info.keys())): continue
                # print(f'> domain: {domain}, date: {str_date}, keyword: {keyword}')
                # sleep(1)

                for word in search_words:

                    if not ord.compare_order_by_list(lev.FOURTH, word, search_words): continue
                    print(f'\n> domain: {domain}, date: {str_date}, keyword: {keyword}, word: {word}')

                    tmp_rows = []

                    # initial task
                    b_info = {}
                    b_info['start_p'] = tmp_p
                    b_info['word'] = word
                    b_info['keyword'] = keyword
                    b_info['domain'] = domain
                    tmp = board.parse(driver, b_info)
                    tmp_rows += que.get_result(tmp)

                    # Taks Queue
                    que.push(que.get_tasks(tmp))
                    ret = que.run(sel, reg)
                    tmp_rows += ret

                    # tmp_rows 사용이유:
                    # que.run 중간에 인터럽트 발생 시
                    # rows에 데이터를 더하지 않는다.
                    # like transaction in DB
                    rows += tmp_rows

                    sleep(1)

            ## loop continue
            tmp_p += timedelta(days=1)
        
    ##############################
    ### close
    ##############################
    if not ord.meta_complete_get():
        # print(f'+++ rows: {rows}')
        res.save_row_part(rows)
        res.save_row_final()

        ord.meta_complete_set(True)
        ord._save_order()

        sel.get_driver().quit()
    else:
        log.info(f'+++ Already collected. {ord._get_order()}')

    return


if __name__ == '__main__':
    main()

