import dotenv
from os import environ,path
p = path.abspath('./.env')
if path.exists(path.abspath('./.env.prod')):
    p = path.abspath('./.env.prod')
    
dotenv.load_dotenv(p,verbose=True,override=True)

AWS_SECRET_ACCESS_KEY = environ['AWS_SECRET_ACCESS_KEY'] if 'AWS_SECRET_ACCESS_KEY' in environ else 'fallback-test-value'
AWS_ACCESS_KEY_ID = environ['AWS_ACCESS_KEY'] if 'AWS_ACCESS_KEY' in environ else 'fallback-test-value'

S3_BUCKET = environ['S3_BUCKET'] if 'S3_BUCKET' in environ else 'fallback-test-value'
MODEL_PATH = environ['MODEL_PATH'] if 'MODEL_PATH' in environ else 'fallback-test-value'
HUD_HOST = environ['DB_HOST'] if 'DB_HOST' in environ else '    '
HUD_DATABASE_NAME = environ['DB_NAME'] if "DB_NAME" in environ else 'DB_NAME'
HUD_LOGIN = environ.get('DB_USER') if 'DB_USER' in environ else 'DB_USER'
HUD_PASSWORD = environ.get('DB_PASSWORD') if 'DB_PASSWORD' in environ else 'DB_PASSWORD'
TMP_EMBEDDINGS_TABLE_NAME = environ.get('TMP_EMBEDDINGS_TABLE_NAME') if 'TMP_EMBEDDINGS_TABLE_NAME' in environ else 'TOPIC_THUNDER_TMP_EMBEDDINGS_TABLE'
SRC_TABLE_NAME = environ.get('ARTICLES_TABLE')
EMBEDDINGS_TABLE = environ.get('EMBEDDINGS_TABLE') # TODO: embeddings table name should be specified based on the model used for embedding generation
TOPIC_LABELS_TABLE = environ.get('TOPIC_LABELS_TABLE')
TOPIC_DESC_TABLE = environ.get('TOPIC_DESC_TABLE')
DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)
