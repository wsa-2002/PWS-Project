from uuid import UUID
from typing import Optional, Sequence
from datetime import datetime

from base import do

from .util import pyformat2psql
from . import pool_handler


async def add(name: str, s3_file_uuid: UUID, filename: str, submit_time: datetime, score: float = None):
    sql, params = pyformat2psql(
        sql=fr"INSERT INTO MLModelInfo"
            fr"            (name, s3_file_uuid, filename, submit_time, score)"
            fr"     VALUES (%(name)s, %(s3_file_uuid)s, %(filename)s, %(submit_time)s, %(score)s)"
            fr"  RETURNING id",
        name=name, s3_file_uuid=s3_file_uuid, filename=filename, submit_time=submit_time, score=score,
    )
    _id, = await pool_handler.pool.fetchrow(sql, *params)
    return _id


async def browse(name: str) -> Sequence[do.MLModelInfo]:
    sql, params = pyformat2psql(
        sql=fr"SELECT id, name, s3_file_uuid, submit_time, score, filename"
            fr"  FROM mlmodelinfo"
            fr" WHERE name = %(name)s"
            fr" ORDER BY id",
        name=name,
    )
    results = await pool_handler.pool.fetch(sql, *params)
    return [do.MLModelInfo(id=id_,
                           name=name,
                           s3_file_uuid=s3_file_uuid,
                           submit_time=submit_time,
                           score=score,
                           filename=filename)
            for id_, name, s3_file_uuid, submit_time, score, filename in results]


async def get(model_id: int) -> do.MLModelInfo:
    sql, params = pyformat2psql(
        sql=fr"SELECT id, name, s3_file_uuid, submit_time, score, filename"
            fr"  FROM mlmodelinfo"
            fr" WHERE id = %(model_id)s",
        model_id=model_id,
    )
    id_, name, s3_file_uuid, submit_time, score, filename = await pool_handler.pool.fetchrow(sql, *params)
    return do.MLModelInfo(
        id=id_,
        name=name,
        s3_file_uuid=s3_file_uuid,
        submit_time=submit_time,
        score=score,
        filename=filename,
    )