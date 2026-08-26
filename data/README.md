# Data

The current prototype was developed against the combined NSL-KDD workbook:

`Copy of DOC-20260825-WA0002.xlsx`

The workbook contains 148,517 records and preserves the original train/test
membership in the `split` column.

Place the workbook in the repository root when running the prototype.

## Important research limitation

NSL-KDD is not timestamped packet-capture telemetry. It is therefore used
as a reproducible prototype dataset for state-window construction. The final
NTRO-grade benchmark should use timestamped telemetry such as CIC-IDS2018,
CTU-13, CICIoT2023 or PCAP-derived features for genuine temporal dynamics.
