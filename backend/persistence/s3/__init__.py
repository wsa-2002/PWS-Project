import boto3
import typing
from uuid import UUID

from base import mcs
from config import S3Config


class S3Handler(metaclass=mcs.Singleton):
    def __init__(self):
        self._session = boto3.Session()
        self._client = None
        self._resource = None
        self._buckets = {}

    def initialize(self, s3_config: S3Config):
        self._client = self._session.client(
            's3',
            endpoint_url=s3_config.endpoint,
            aws_access_key_id=s3_config.access_key,
            aws_secret_access_key=s3_config.secret_key,
        )

        self._resource = self._session.resource(
            's3',
            endpoint_url=s3_config.endpoint,
            aws_access_key_id=s3_config.access_key,
            aws_secret_access_key=s3_config.secret_key,
        )

    def close(self):
        self._client.close()
        self._resource.close()

    def get_bucket(self, bucket_name):
        """
        If the bucket requested is not yet created, will create the bucket.
        """
        try:
            return self._buckets[bucket_name]
        except KeyError:
            bucket = self.create_bucket(bucket_name)
            self._buckets[bucket_name] = bucket
            return bucket

    def create_bucket(self, bucket_name):
        return self._resource.Bucket(bucket_name)

    def sign_url(self, bucket: str, key: str, filename: str) -> str:
        return self._client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket,
                'Key': key,
                'ResponseContentDisposition': f"attachment; filename={filename};"
            },
            ExpiresIn=3600,
        )

    def upload(self, file: typing.IO, key: UUID, bucket_name: str = 'temp'):
        bucket = self._resource.Bucket(bucket_name)
        bucket.upload_fileobj(file, str(key))


s3_handler = S3Handler()
