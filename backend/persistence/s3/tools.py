import typing
from uuid import UUID
from datetime import datetime

import log
from base import do

from . import s3_handler


def sign_url(bucket: str, key: str, filename: str, expire_secs: int = 86400) \
        -> str:
    start_time = datetime.now()
    log.info(f'Start getting S3 file sign url ...')

    signed_url = s3_handler.sign_url(
        bucket=bucket,
        key=key,
        filename=filename,
    )

    exec_time_ms = (datetime.now() - start_time).total_seconds() * 1000
    log.info(f'Ended get S3 file {sign_url=} after {exec_time_ms} ms')

    return signed_url


def sign_url_from_do(s3_file: do.S3File, filename: str, expire_secs: int = 86400) -> str:
    return sign_url(
        bucket=s3_file.bucket,
        key=s3_file.key,
        filename=filename,
        expire_secs=expire_secs,
    )


def upload(bucket_name: str, file: typing.IO, file_uuid: UUID) -> do.S3File:
    """
    :return: do.S3File
    """
    start_time = datetime.now()
    log.info(f'Starting S3 file upload: {bucket_name=}, {file_uuid=}')

    bucket = s3_handler.get_bucket(bucket_name)
    key = str(file_uuid)
    bucket.upload_fileobj(file, key)

    exec_time_ms = (datetime.now() - start_time).total_seconds() * 1000
    log.info(f'Ended S3 file upload after {exec_time_ms} ms')

    return do.S3File(uuid=file_uuid, bucket=bucket_name, key=key)
