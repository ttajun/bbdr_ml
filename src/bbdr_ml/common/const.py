
class Const:

    SITE_NAVER_NEWS = 'naverNews'

    CATEGORY_LIST = ['blog', 'cafe', 'community', 'news', 'shoppingmall', 'sns']
    SITE_LIST= [
        '82coo', 'auction', 'bigki', 'bobae','clien', 
        'coupang', 'd-blo', 'd-caf', 'daumNews', 'dcins',
        'fmkorea', 'g-market', 'hygall', 'mlbpark', 'n-blo',
        'n-caf', 'n-shopping', 'naverNews', 'naver_shopping', 'pann',
        'ruliweb', 'tdgall', 'twitter', 'ygosu', 'youtu',
        'yuldo', '11st', 'auction'
    ]

    DOMAIN = {

        'massagechair': {
            '바디프랜드': ['바디프랜드 안마의자'],
            '코지마': ['코지마 안마의자'],
            '휴테크': ['휴테크 안마의자'],
            # '힐링미': ['힐링미 안마의자'],
            # '웰모아': ['웰모아 안마의자'],
            '제스파': ['제스파 안마의자'],
            # '케어렉스': ['케어렉스 안마의자'],
            '세라젬': ['세라젬 안마의자'],
            # '브람스': ['브람스 안마의자'],
            # '쿠쿠': ['쿠쿠 안마의자'],
            # '누하스': ['누하스 안마의자'],
            # '슈퍼체어': ['슈퍼체어 안마의자'],
            # 'SK매직': ['SK매직 안마의자'],
            # '장수헬스케어': ['장수헬스케어 안마의자'],
            # '비욘드릴렉스': ['비욘드릴렉스 안마의자'],
        },

        'probio': {
            '듀오락': ['듀오락'],
            '락토핏': ['락토핏'],
            '바이오락토': ['바이오락토'],
            '밸런스위드인': ['밸런스위드인'],
            '락피도': ['락피도'],
            '장안에화제': ['장안에화제'],
            '중외제약울트라': ['중외제약울트라 유산균'],
            '드시모네': ['드시모네'],
            '하루웰빙': ['하루웰빙 유산균'],
            '에스더포뮬러': ['에스더포뮬러'],
            '이너플로라': ['이너플로라'],
            '상아제약': ['상아제약 유산균'],
            '내츄럴플러스': ['내츄럴플러스 유산균'],
            '지큐랩': ['지큐랩'],
            '퍼펙트바이오틱스': ['퍼펙트 바이오틱스'],
            '한미': ['한미 유산균'],
            'BYO바이오생유산균': ['BYO바이오생 유산균'],
            '락토바이옴': ['락토바이옴'],
            '덴프스': ['덴프스'],
            '비비랩': ['비비랩 유산균'],
            '휴럼트루락': ['휴럼트루락'],
            '엘레나': ['엘레나 유산균'],
            '일양약품신바이오틱스': ['일양약품 신바이오틱스'],
            '이너리스펙타': ['이너리스펙타'],
            '비오비타': ['비오비타'],
            '셀티바': ['셀티바 유산균'],
            'GNM': ['GNM 유산균'],
            '하이락비피더스': ['하이락 비피더스'],
            '메가초유6H': ['메가초유6H'],
            '락티브': ['락티브'],
            '오한진유산균': ['오한진 유산균'],
            '바료랑유산균': ['바료랑 유산균'],
            '아임비오유산균': ['아임비오 유산균'],
            '프로스랩유산균': ['프로스랩 유산균'],
            '닥터릴틴유산균': ['닥터릴틴 유산균'],
        },

        'womenprobio': {
            '엘레나': ['엘레나 유산균'],
            '비비랩': ['비비랩 유산균'],
            '이너플로라': ['이너플로라'],
            # '이너리스펙타': ['이너리스펙타'],
            # '프로스랩핑크': ['프로스랩핑크'],
            '지노마스터': ['지노마스터'],
            # '시크릿케어': ['락티브 시크릿케어'],
            # '트루락이브': ['트루락이브'],
            # '지노프레쉬': ['지노프레쉬'],
            # '여에스더화이트': ['여에스더 화이트'],
        }

    }


    DEFAULT_CSV_FIELDS = [
        'uid','url','title','c_cnt','source_text',
        'content','content2', 'category','source_site','comment', 
        'r_cnt', 'i_date', 'bad', 'tag', 'script', 
        'c_date', 'domain'
        ]

    DEFAULT_DW_FIELDS = [
        'uid','url','title','c_cnt','source_text',
        'content', 'category','source_site','comment', 
        'r_cnt', 'i_date', 'bad', 'tag', 'script', 
        'c_date', 'domain'
    ]

    CSV_FIELD_TYPE = {
        'uid':'str',
        'url':'str',
        'title':'str',
        'c_cnt':'str',
        'source_text':'str',
        'content':'str',
        'content2':'str',
        'category':'str',
        'source_site':'str',
        'comment':'str',
        'r_cnt':'str',
        'i_date':'str', 
        'bad':'str',
        'tag':'str',
        'script':'str',
        'c_date':'str',
        'domain':'str',
    }

    SSH_HOST = '192.168.0.103'
    SSH_PORT = '2200'
    SSH_USER = 'root'
    SSH_PASSWORD = 'bigidean1!'
    SSH_DEST_PATH = '/BBDR/collect/raw'

