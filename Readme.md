## Results

This directory contains all experimental outputs used in the evaluation of the models.  
The results are organized to ensure full transparency, reproducibility, and extensibility, allowing external users to recompute metrics or define new evaluation criteria directly from raw predictions.

All reported metrics in the paper are derived exclusively from the contents of this folder.

---

### Directory overview

The `results/` directory contains the following subfolders:

- `Global metrics/`  
- `Metrics by company/`  
- `Predictions per stock by each model/`


---

### Predictions per stock by each model

This folder contains the raw numerical outputs of the experiments and represents the lowest level of aggregation.

For each model, stock, and partition, it includes:
- Model predictions
- Corresponding ground truth values
- Metadata describing each evaluation partition (e.g. time and emissions)

Predictions are stored per stock and per partition, enabling:
- Reproduction of all reported metrics
- Computation of alternative or custom metrics
- Fine-grained error analysis at partition or stock level

This folder should be considered the source of truth for all downstream evaluations.

---

### Metrics by company

This folder provides stock-level aggregated results.

For each stock and each model, it includes:
- Aggregated metadata across the five evaluation partitions
- Metrics computed independently for each partition
- Average metrics across the five partitions for that stock

These results are intended for:
- Per-stock performance analysis
- Comparing model behavior across different stocks
- Inspecting variability across partitions

---

### Global metrics

This folder contains dataset-level aggregated results.

Metrics are obtained by averaging results across the 35 stocks.  
Each value represents the mean performance of a model over all stocks.

This is the primary reference for global model comparison.

---

### Reproducibility and metric computation

All aggregated metrics provided in this repository are computed solely from the raw prediction files contained in the Predictions per stock by each model folder.

No external processing steps, filtering, or manual aggregation are applied outside the data provided here.

Users can fully reproduce the reported results or define new metrics using only the contents of this directory.


## Source code

This directory contains the code used to run the experiments and generate the results reported in the paper.

The folder includes:
- The training and execution scripts used in the experimental pipeline
- The dataset file used in the experiments
- The environment specification required to reproduce the results

### Contents

src/
- IBEX-FinTime.csv  
  Dataset used for training and evaluation.

- test.py  
  Main script used to run experiments and generate predictions and metrics.

- requirements.txt  
  Python dependencies corresponding to the environment used in the experiments.

All experiments reported in the paper can be reproduced using the code and configuration provided in this directory, together with the outputs stored in the `results/` folder.
