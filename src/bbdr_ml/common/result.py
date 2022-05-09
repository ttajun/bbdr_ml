
import os

import pandas as pd

from common import util, logger, order, const
log = logger.make_logger(__name__)


class Result:

    def __init__(self, order:order.Order):
        root_dir = util.get_program_path()
        result_dir = f'{root_dir}/result'
        util.create_directory(result_dir)

        self.result_dir = result_dir
        self.order = order


    def get_result_dir(self):
        return self.result_dir

    
    def save_row_part(self, rows):

        if len(rows) == 0: return

        ord = self.order
        base = ord.args_basename_get()
        part = ord.meta_partno_get()
        file = f'{base}.part{part}'
        full = f'{self.get_result_dir()}/{file}'
        print('*'*100)
        # print(f'base: {base}, part: {part}, file: {file}')
        # print(f'rows: {rows}')

        df = pd.DataFrame(rows, columns=const.Const.DEFAULT_CSV_FIELDS)

        field = ['title', 'content', 'content2', 'comment', 'script', 'tag']
        df[field] = df[field].applymap(util.remove_comma)
        df[field] = df[field].applymap(util.remove_newline)

        log.info(df[['source_site', 'c_date', 'source_text', 'tag', 'r_cnt', 'title', 'content']].applymap(lambda x: util.short_str(x, 30)))
        log.info(df['source_text'].value_counts())
        print('*'*100)
        df_size = len(df)
        # print(f'size: {df_size}')

        if df_size > 0:
            df.to_csv(full, index=False, header=False, encoding='utf-8-sig')
            part += 1
            ord.meta_partno_set(part)
            ord.meta_partcsv_append(file)

    
    def save_row_final(self):
        os.chdir(self.get_result_dir())
        ord = self.order
        con = const.Const

        # merge part files
        csv = ord.meta_partcsv_get()
        base = ord.args_basename_get()
        try:
            df = pd.concat(map(lambda x : pd.read_csv(x, names=con.DEFAULT_CSV_FIELDS, dtype=con.CSV_FIELD_TYPE, encoding='utf-8'), csv), ignore_index=True)
        except Exception as e:
            log.error(f'{e}')
            log.error(f'> csv: {csv}')
            exit(1)

        ### split by domain
        df:pd.DataFrame
        log.info(f'> row size: {len(df)}')
        df = df.drop_duplicates(subset = ['uid'], keep='last')
        log.info(f'> row size duplicates: {len(df)}')

        domains = df['domain'].drop_duplicates().to_list()
        log.info(f'> domains: {domains}')
        for d in domains:
            df_d = df[df['domain'] == d]
            file = f'{base}.csv'
            df_d.to_csv(file, index=False, header=False, encoding='utf-8-sig')
            log.info(f'>> {d}({len(df_d)}): {file}')

