CREATE TABLE s3_file (
  uuid      UUID    PRIMARY KEY,
  bucket  VARCHAR NOT NULL,
  key     VARCHAR NOT NULL
);

CREATE TABLE sheet (
  id            SERIAL   PRIMARY KEY,
  url           VARCHAR  NOT NULL,
  s3_file_uuid  UUID     NOT NULL REFERENCES s3_file(uuid)
);
