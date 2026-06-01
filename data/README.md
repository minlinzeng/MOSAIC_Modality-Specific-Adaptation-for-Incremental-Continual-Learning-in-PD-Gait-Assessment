# Data directory

Raw and processed datasets are **not** tracked in git. Place files here following the layout described in the root [README.md](../README.md#data-preparation).

Override the project root if needed:

```bash
export JBHI26_ROOT=/path/to/JBHI26
```

Expected layout:

```
data/
├── weargait/HC_raw/  PD_raw/
├── fog/C3Dfiles/  IMU/  predictions/  PDFEinfo.xlsx  data_cache/fog_processed/
└── fbg/C3Dfiles/  Processed_FBG_Manifolds/
```
