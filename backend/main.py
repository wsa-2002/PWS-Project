
with open('logging.yaml', 'r') as file:
    import yaml
    log_config = yaml.safe_load(file.read())

    import logging.config
    logging.config.dictConfig(log_config)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import app_config

app = FastAPI(
    title=app_config.title,
    docs_url=app_config.docs_url,
    redoc_url=app_config.redoc_url,
)

origins = [
    'http://localhost',
    'http://localhost:3000',
    'http://localhost:3006',
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event('startup')
async def app_startup():
    from config import db_config
    from persistence.database import pool_handler
    await pool_handler.initialize(db_config=db_config)

    # # if s3 needed
    from config import s3_config
    from persistence.s3 import s3_handler
    s3_handler.initialize(s3_config=s3_config)


@app.on_event('shutdown')
async def app_shutdown():
    from persistence.database import pool_handler
    await pool_handler.close()
    #
    # if s3 needed
    from persistence.s3 import s3_handler
    s3_handler.close()

import processor.http
processor.http.register_routers(app)
