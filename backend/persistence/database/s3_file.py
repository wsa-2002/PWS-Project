from uuid import UUID

from base import do

from .util import pyformat2psql
from . import pool_handler


async def add(s3_file: do.S3File) -> None:
    sql, params = pyformat2psql(
        sql=fr"INSERT INTO s3_file"
            fr"            (uuid, key, bucket)"
            fr"     VALUES (%(uuid)s, %(key)s, %(bucket)s)",
        uuid=s3_file.uuid, key=s3_file.key, bucket=s3_file.bucket,
    )
    await pool_handler.pool.execute(sql, *params)


async def get(s3_file_uuid: UUID) -> do.S3File:
    sql, params = pyformat2psql(
        sql=fr"SELECT uuid, bucket, key"
            fr"  FROM s3_file"
            fr" WHERE uuid = %(s3_file_uuid)s",
        s3_file_uuid=s3_file_uuid,
    )
    uuid, bucket, key = await pool_handler.pool.fetchrow(sql, *params)
    return do.S3File(uuid=uuid, bucket=bucket, key=key)
