from base import do

from .util import pyformat2psql
from . import pool_handler


async def get_config(config_id: int = 1) -> do.Config:
    sql, params = pyformat2psql(
        sql=fr"SELECT id, extract_interval, identify_threshold"
            fr"  FROM config"
            fr" WHERE id = %(config_id)s",  # TODO: May have more config
        config_id=config_id,
    )
    _id, extract_interval, identify_threshold = await pool_handler.pool.fetchrow(sql, *params)
    return do.Config(
        id=_id,
        extract_interval=extract_interval,
        identify_threshold=identify_threshold,
    )
