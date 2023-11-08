"""
data objects
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass
class S3File:
    uuid: UUID
    key: str
    bucket: str


@dataclass
class Sheet:
    id: int
    url: str
    filename: str
    s3_file_uuid: UUID


@dataclass
class Config:
    id: int
    extract_interval: float
    identify_threshold: float


@dataclass
class MLModelInfo:
    id: int
    name: str
    s3_file_uuid: UUID
    submit_time: datetime
    filename: str
    score: Optional[float]

