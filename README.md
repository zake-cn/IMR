# IMR IECE Demo

This demo is scoped to the IECE paper presentation workflow. The default
dataset is:

```text
data/Implicit_emotion_cause_dataset.xml
```

## 1. Set Up The Environment

```bash
uv sync
```

On the first run, the training and search entry points automatically precompute
embeddings if they are missing. You can also run the precomputation manually:

```bash
uv run python -m src.prepare_embeddings
```

## 2. Search For Best Hyperparameters

```bash
uv run python -m src.hparam_search --n-trials 50 --storage sqlite:///optuna_iece.db --output logs/best_hparams.json
```

Run a quick smoke test to verify that the workflow can execute:

```bash
uv run python -m src.hparam_search --smoke-test --n-trials 2 --n-smoke-epochs 1 --output logs/best_hparams.json
```

## 3. Train With Best Hyperparameters

```bash
uv run python -m src.train
```

`src.train` looks for `logs/best_hparams.json` by default:

- If the file exists, it loads the best hyperparameters before training.
- If the file does not exist, it trains with the defaults in `src/config.py`.

You can also specify a hyperparameter file or skip existing hyperparameters:

```bash
uv run python -m src.train --hparams-file logs/best_hparams.json
uv run python -m src.train --no-hparams
```

## 4. Outputs

Training logs are saved to `logs/training_*.jsonl`. Each epoch includes:

- `train_loss` (joint ER + IECE optimization loss)
- `val_loss` (joint ER + IECE optimization loss)
- `val_precision`
- `val_recall`
- `val_f1`

The best model for each fold is saved to `checkpoints/fold_*_best.pt`.
