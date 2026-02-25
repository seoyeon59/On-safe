import pymysql
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# 사업화를 고려한 설정 정보 관리
DB_SETTINGS = {
    'host': 'localhost',
    'user': 'root',
    'password': 'bear0205!@!@',
    'db': 'OnSafe',
    'port': 3306,
    'charset': 'utf8mb4'
}

# 1. SQLAlchemy 엔진 (대량 분석용)
password_encoded = quote_plus(DB_SETTINGS['password'])
sqlalchemy_url = (
    f"mysql+pymysql://{DB_SETTINGS['user']}:{password_encoded}@"
    f"{DB_SETTINGS['host']}:{DB_SETTINGS['port']}/{DB_SETTINGS['db']}?charset={DB_SETTINGS['charset']}"
)
engine = create_engine(sqlalchemy_url, pool_recycle=3600, pool_size=10)

# 2. PyMySQL 연결 함수 (일반 CRUD용)
def get_db_connection():
    return pymysql.connect(
        host=DB_SETTINGS['host'],
        user=DB_SETTINGS['user'],
        password=DB_SETTINGS['password'],
        database=DB_SETTINGS['db'],
        port=DB_SETTINGS['port'],
        charset=DB_SETTINGS['charset'],
        cursorclass=pymysql.cursors.DictCursor
    )