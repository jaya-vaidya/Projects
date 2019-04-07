CREATE  INDEX idx_pat_pat_id
ON PATIENT (PATIENT_ID);

CREATE  INDEX idx_doc_doc_id
ON DOCTOR (DOC_ID);

CREATE INDEX idx_bill_bill_dt
ON BILL (BILL_DATE);
