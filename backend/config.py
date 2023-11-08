import os

from dotenv import dotenv_values

env_values = {
    **dotenv_values(".env"),
    **os.environ,
}


class DBConfig:
    host = env_values.get('PG_HOST')
    port = env_values.get('PG_PORT')
    username = env_values.get('PG_USERNAME')
    password = env_values.get('PG_PASSWORD')
    db_name = env_values.get('PG_DBNAME')
    max_pool_size = int(env_values.get('PG_MAX_POOL_SIZE'))


class AppConfig:
    title = env_values.get('APP_TITLE')
    docs_url = env_values.get('APP_DOCS_URL', None)
    redoc_url = env_values.get('APP_REDOC_URL', None)


class S3Config:
    endpoint = env_values.get('S3_ENDPOINT')
    access_key = env_values.get('S3_ACCESS_KEY')
    secret_key = env_values.get('S3_SECRET_KEY')


class LineConfig:
    channel_access_token = env_values.get('LINE_CHANNEL_ACCESS_TOKEN')
    channel_secret = env_values.get('LINE_CHANNEL_SECRET')


db_config = DBConfig()
app_config = AppConfig()
s3_config = S3Config()
line_config = LineConfig()
