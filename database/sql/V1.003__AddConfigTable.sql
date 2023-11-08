CREATE TABLE config (
    id                    SERIAL  PRIMARY KEY,
    extract_interval      FLOAT   NOT NULL,
    identify_threshold    FLOAT   NOT NULL
);

INSERT INTO config values (1, 0.5, 0.07);
