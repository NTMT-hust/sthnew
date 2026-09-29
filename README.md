# DRIAMS hybrid spectral and graph model

The project is organized into `data/` (DRIAMS loading and site splits),
`models/` (spectrum autoencoder and hybrid heterogeneous graph network), and
`train.py` (end-to-end training, validation, test evaluation, and output saving).

## Requested site split

- Train: DRIAMS-A
- Validation and model selection: DRIAMS-D + DRIAMS-C
- Final test: DRIAMS-B

The default dataset directory resolves to
`C:\Nguyen Tri\Paper\vi_khuan_khang_thuoc\DRIAMS`.

## Run

Install dependencies and run the full training pipeline:

```powershell
python -m pip install -r requirements.txt
python train.py
```

Or pass the dataset path explicitly:

```powershell
python train.py --driams-root "C:\Nguyen Tri\Paper\vi_khuan_khang_thuoc\DRIAMS"
```

The first stage trains an unlabeled spectrum autoencoder using DRIAMS-A only.
It exports spectrum embeddings for A, B, C, and D. The AST model then trains
on labeled isolate-antibiotic edges from A, chooses the epoch with the best
combined B/C macro-F1, and evaluates once on D. Target AST edges in each
training minibatch are withheld from graph message passing.

The code adapts HGnnDTI's low-level structure encoder and high-level
heterogeneous interaction graph to isolate-antibiotic susceptibility. It is an
AST classifier rather than a drug-target interaction predictor. Susceptibility
classes are S and R. Intermediate (I), missing, and other non-binary values
are treated as unlabeled and excluded from AST training and evaluation.

## Drug structures

`antibiotic_smiles.csv` contains structures for 22 antibiotics, including
oxacillin and clindamycin. The model uses matching AST columns and listed name
aliases. Unsupported drug columns are excluded and listed in the metrics file.
Add valid SMILES rows to extend coverage.

## Saved results

The default output directory is `outputs/site_split/`:

- `spectrum_encoder.pt`: trained autoencoder and training-set normalization.
- `isolate_embeddings.npz`: embeddings and aligned sample/site/species/year IDs.
- `hybrid_ast_best.pt`: best model selected on B/C validation macro-F1.
- `metrics.json`: split sizes, all aggregate metrics by D+C, D, C, and B, and
  training histories, with complete nested metrics per antibiotic and per
  bacterial species for validation and test.
- `metrics_per_drug.csv` and `metrics_per_bacteria.csv`: one row per drug or
  species and split, including sample count, classification metrics, confusion
  matrix, and class report.
- `training_history.csv` and `representation_history.csv`.
- `validation_DC_predictions.csv` and `test_B_predictions.csv`.
- `splits.csv`: isolate-level site assignment.

Metrics include accuracy, balanced accuracy, macro/weighted/micro precision,
recall and F1, Hamming loss, Cohen's kappa, Matthews correlation, log loss,
ROC AUC, average precision, confusion matrix, and class
report. AUC metrics are `null` if a split lacks a required class.
