ALTER TABLE sheet
  ADD COLUMN filename VARCHAR;

UPDATE sheet
   SET filename = url;

ALTER TABLE sheet ALTER COLUMN filename set not null;