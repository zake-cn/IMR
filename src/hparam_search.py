"""
Bayesian hyperparameter search with Optuna TPE.
=====================================
Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for Bayesian
optimization and maximizes validation Cause F1.

Best-parameter output
---------------------
After the search, the best parameters are saved automatically as JSON.
Default path: logs/best_hparams.json.

Example payload:
  {
    "best_trial": 12,
    "best_f1": 0.7234,
    "params": { "learning_rate": 1.5e-4, "d_model": 256, ... },
    "derived": { "nhead": 8, "dim_ff": 512 }
  }

If --storage is specified, for example sqlite:///optuna.db, the full trial
history is persisted to that database. You can inspect it with
optuna-dashboard after installing it:
  optuna-dashboard sqlite:///optuna.db

Search strategy
---------------
- High-priority parameters: learning_rate, dropout, d_model.
- Medium-priority parameters: layer counts, dim_ff, imr_iterations, batch_size.
- Low-priority parameters: weight_decay, label_smoothing, warmup_epochs.

Constraints
-----------
- DL_NHEAD is derived from DL_D_MODEL to ensure divisibility.
- DL_DIM_FF is constrained to an integer multiple of DL_D_MODEL.
- Smoke-test mode runs only one fold for N_SMOKE_EPOCHS epochs per trial.

Usage
-----
    # Full search, 50+ trials recommended.
    python -m src.hparam_search --n-trials 50 --storage "sqlite:///optuna_smoke.db"

    # Smoke-test mode to verify the searcher can run.
    python -m src.hparam_search --smoke-test --n-trials 2 --storage "sqlite:///optuna_smoke.db"
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging

import torch

import optuna
from optuna.samplers import TPESampler

from src.config import Config
from src.dataset import load_all_data
from src.model import IMRModel
from src.train import train

logger = logging.getLogger(__name__)

# ============================================================================
# Hyperparameter search space.
# ============================================================================

# Grouped by priority; order is an importance hint.
SEARCH_SPACE = {
    # --- High priority: core training dynamics ---
    "learning_rate": {
        "type": "float",
        "low": 5e-5,
        "high": 5e-4,
        "log": True,  # Log-uniform distribution, suitable for LR.
    },
    "lr_shared": {
        "type": "float",
        "low": 0.5,
        "high": 1.5,
        "step": 0.25,
    },
    "lr_emo": {
        "type": "float",
        "low": 0.5,
        "high": 2.0,
        "step": 0.25,
    },
    "lr_cause": {
        "type": "float",
        "low": 0.5,
        "high": 2.0,
        "step": 0.25,
    },
    "dropout": {
        "type": "float",
        "low": 0.1,
        "high": 0.5,
        "step": 0.05,
    },
    "d_model": {
        "type": "categorical",
        "choices": [128, 192, 256, 320],  # Must be divisible by nhead.
    },
    # --- Medium priority: model structure ---
    "n_emotion_layers": {
        "type": "int",
        "low": 1,
        "high": 5,
    },
    "n_cause_layers": {
        "type": "int",
        "low": 1,
        "high": 4,
    },
    "dim_ff_multiplier": {
        "type": "categorical",
        "choices": [2, 3, 4],  # dim_ff = d_model * multiplier.
    },
    "imr_iterations": {
        "type": "int",
        "low": 1,
        "high": 5,
    },
    "batch_size": {
        "type": "categorical",
        "choices": [32, 48, 64, 96],
    },
    # --- Low priority: regularization and scheduling ---
    "weight_decay": {
        "type": "float",
        "low": 1e-4,
        "high": 0.1,
        "log": True,
    },
    "label_smoothing": {
        "type": "float",
        "low": 0.0,
        "high": 0.2,
        "step": 0.05,
    },
    "warmup_epochs": {
        "type": "int",
        "low": 1,
        "high": 5,
    },
}


def _get_nhead(d_model: int) -> int:
    """Derive the largest valid nhead up to 8 that divides d_model."""
    for nhead in [8, 4, 2, 1]:
        if d_model % nhead == 0:
            return nhead
    return 1


def _suggest_params(trial: optuna.Trial) -> dict:
    """Sample all hyperparameters from a trial and return the full parameter dict."""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        t = spec["type"]
        if t == "float":
            kwargs = {"low": spec["low"], "high": spec["high"]}
            if spec.get("log"):
                kwargs["log"] = True
            if spec.get("step") and not spec.get("log"):
                kwargs["step"] = spec["step"]
            params[name] = trial.suggest_float(name, **kwargs)
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


# ============================================================================
# Optuna objective.
# ============================================================================


def objective(
    trial: optuna.Trial,
    smoke_test: bool = False,
    n_smoke_epochs: int = 2,
) -> float:
    """
    Objective function for one trial.

    Evaluation strategy:
    - Normal mode: run only fold 1 for up to DL_EPOCHS with early stopping,
      then return validation Cause F1.
    - Smoke mode: run only fold 1 for up to n_smoke_epochs to verify viability.

    Memory management:
    - Large objects such as model and folds are explicitly deleted in finally,
      then empty_cache() is called to avoid trial-to-trial memory accumulation.

    Returns:
        val_f1: validation Cause F1; higher is better.
    """
    import gc

    params = _suggest_params(trial)

    # Derive dependent parameters.
    d_model: int = params["d_model"]
    nhead: int = _get_nhead(d_model)
    dim_ff: int = d_model * params["dim_ff_multiplier"]
    epochs: int = n_smoke_epochs if smoke_test else Config.DL_EPOCHS
    batch_size: int = params["batch_size"]

    logger.info(
        f"[Trial {trial.number}] params={params}, nhead={nhead}, dim_ff={dim_ff}"
    )

    device = Config.DEVICE

    # Temporarily override Config. This is not thread-safe, but works in one process.
    _override_config(params, nhead, dim_ff)

    # Initialize as None so finally can safely delete these names.
    model = None
    folds = None
    train_loader = val_loader = labeled_loader = None

    try:
        folds = load_all_data(batch_size=batch_size)
        train_loader, val_loader, labeled_loader = folds[0]

        model = IMRModel(
            hidden_size=Config.DL_HIDDEN_SIZE,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            n_emotion_layers=params["n_emotion_layers"],
            n_cause_layers=params["n_cause_layers"],
            num_emotions=Config.get_num_emotions(),
            n_iterations=params["imr_iterations"],
            dropout=params["dropout"],
            use_event_emb=Config.DL_USE_EVENT_EMB,
        ).to(device)

        _, best_f1, _ = train(
            model=model,
            train_loader=labeled_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=params["learning_rate"],
            device=device,
            patience=Config.DL_EARLY_STOP_PATIENCE if not smoke_test else 999,
        )

        # Report intermediate value for Optuna pruning support.
        trial.report(best_f1, step=0)
        return best_f1

    except Exception as e:
        logger.warning(f"[Trial {trial.number}] encountered an exception: {e}")
        raise optuna.exceptions.TrialPruned() from e
    finally:
        # === Explicitly release memory to avoid accumulation across trials. ===
        # 1. Delete model and DataLoader references.
        del model
        del folds, train_loader, val_loader, labeled_loader
        # 2. Force Python GC immediately instead of waiting for the next cycle.
        gc.collect()
        # 3. Ask the CUDA caching allocator to return freed blocks to its pool.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _restore_config()


# ============================================================================
# Temporary Config overrides, not thread-safe and intended for one process.
# ============================================================================

_config_backup: dict = {}


def _override_config(params: dict, nhead: int, dim_ff: int) -> None:
    """Temporarily write sampled hyperparameters into Config and save originals."""
    global _config_backup
    overrides = {
        "DL_D_MODEL": params["d_model"],
        "DL_NHEAD": nhead,
        "DL_DIM_FF": dim_ff,
        "DL_N_EMOTION_LAYERS": params["n_emotion_layers"],
        "DL_N_CAUSE_LAYERS": params["n_cause_layers"],
        "DL_DROPOUT": params["dropout"],
        "DL_IMR_ITERATIONS": params["imr_iterations"],
        "LR_SHARED": params["lr_shared"],
        "LR_EMO": params["lr_emo"],
        "LR_CAUSE": params["lr_cause"],
        "DL_WEIGHT_DECAY": params["weight_decay"],
        "DL_LABEL_SMOOTHING": params["label_smoothing"],
        "DL_WARMUP_EPOCHS": params["warmup_epochs"],
    }
    _config_backup = {k: getattr(Config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(Config, k, v)


def _restore_config() -> None:
    """Restore Config values saved before the trial."""
    for k, v in _config_backup.items():
        setattr(Config, k, v)


# ============================================================================
# Main search entry point.
# ============================================================================


def run_search(
    n_trials: int = 50,
    study_name: str = "imr_hparam_search",
    storage: str | None = None,
    smoke_test: bool = False,
    n_smoke_epochs: int = 2,
    output: str | Path = "logs/best_hparams.json",
) -> optuna.Study:
    """
    Start Bayesian hyperparameter search.

    Args:
        n_trials:        Number of trials to run; 50 or more is recommended.
        study_name:      Optuna study name.
        storage:         Persistent storage URL, such as sqlite:///optuna.db.
        smoke_test:      Whether to run smoke-test mode.
        n_smoke_epochs:  Maximum epochs per fold in smoke-test mode.
        output:          Path for the best-parameter JSON output.

    Returns:
        optuna.Study object containing all trial results.
    """
    sampler = TPESampler(
        n_startup_trials=10,  # Randomly explore the first 10 trials.
        multivariate=True,  # Jointly model all parameters to capture correlations.
        seed=Config.RANDOM,
    )

    study = optuna.create_study(
        direction="maximize",  # Maximize Cause F1.
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,  # Allow resume from an existing study.
    )

    mode_tag = "[Smoke Test]" if smoke_test else "[Full Search]"
    Config.apply_dataset_type("IECE")
    logger.info(
        f"{mode_tag} Bayesian hyperparameter search started | trials={n_trials} | "
        f"study={study_name} | storage={storage or 'in-memory'} | dataset={Config.DATASET_TYPE}"
    )

    study.optimize(
        lambda trial: objective(
            trial, smoke_test=smoke_test, n_smoke_epochs=n_smoke_epochs
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # === Report the best result. ===
    best = study.best_trial
    d_model = best.params["d_model"]
    nhead = _get_nhead(d_model)
    dim_ff = d_model * best.params["dim_ff_multiplier"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Search complete. Best Trial #{best.number}")
    logger.info(f"  Best Cause F1 = {best.value:.4f}")
    logger.info(f"  Best Params:")
    for k, v in best.params.items():
        logger.info(f"    {k}: {v}")
    logger.info(f"  [Derived] nhead={nhead}, dim_ff={dim_ff}")
    logger.info(f"{'='*60}")

    # === Save best parameters to JSON. ===
    import json

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_result = {
        "best_trial": best.number,
        "best_f1": best.value,
        "params": best.params,
        "derived": {
            "nhead": nhead,
            "dim_ff": dim_ff,
        },
        "study_name": study_name,
        "storage": storage,
        "n_trials_completed": len(study.trials),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Best parameters saved to: {output_path.resolve()}")

    # Also export full trial history for later analysis.
    trials_output = output_path.parent / "hparam_trials.json"
    trials_payload = []
    for t in study.trials:
        if t.value is None:
            continue
        trials_payload.append(
            {
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                "params": t.params,
            }
        )
    with open(trials_output, "w", encoding="utf-8") as f:
        json.dump(trials_payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Trial history saved to: {trials_output.resolve()}")

    return study


# ============================================================================
# CLI entry point.
# ============================================================================


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress Optuna internal debug logs.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(description="IMR Bayesian hyperparameter search")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of search trials, default 50",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="imr_hparam_search",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna persistent storage URL, such as sqlite:///optuna.db",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/best_hparams.json",
        help="Best-parameter JSON output path, default logs/best_hparams.json",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Smoke-test mode: run only 2 epochs per trial to verify the searcher",
    )
    parser.add_argument(
        "--n-smoke-epochs",
        type=int,
        default=2,
        help="Maximum epochs per fold in smoke-test mode, default 2",
    )
    args = parser.parse_args()

    # Ensure embeddings have been precomputed.
    emb_dir = Path(Config.DL_EMBEDDING_DIR)
    legacy_dir = Path(Config.DL_EMBEDDING_ROOT)
    has_embeddings = (emb_dir / "labeled_text.npy").exists() or (
        legacy_dir / "labeled_text.npy"
    ).exists()

    if not has_embeddings:
        from src.prepare_embeddings import pre_embed

        pre_embed()

    run_search(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        smoke_test=args.smoke_test,
        n_smoke_epochs=args.n_smoke_epochs,
        output=args.output,
    )


if __name__ == "__main__":
    main()
