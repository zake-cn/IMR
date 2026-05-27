"""
IMR demo configuration.

The paper demo is intentionally scoped to the IECE task and the original
Implicit Emotion Cause dataset.
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    RANDOM: int = 2025

    # ---------------------------
    # Dataset and label space.
    # ---------------------------
    DATASET_TYPE: str = "IECE"

    IECE_DATASET: str = os.getenv(
        "IECE_DATASET", "data/Implicit_emotion_cause_dataset.xml"
    )

    IECE_LABELS_MAP_E2C: dict = {
        "Anger": "Anger",
        "Disgust": "Disgust",
        "Fear": "Fear",
        "Guilt": "Guilt",
        "Joy": "Joy",
        "Sadness": "Sadness",
        "Shame": "Shame",
    }
    # Emotion-to-index mapping, fixed by dictionary insertion order.
    IECE_EMO_DICT: dict = {k: i for i, k in enumerate(IECE_LABELS_MAP_E2C.keys())}

    LABELS_MAP_E2C: dict = IECE_LABELS_MAP_E2C
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET: str = IECE_DATASET
    NUM_EMOTION_CATEGORIES: int = len(IECE_EMO_DICT)

    # --- Deep learning hyperparameters ---
    # PLM configuration: used only for static embedding extraction, not trained.
    # EMBEDING_MODEL: str = "dienstag/chinese-roberta-wwm-ext"  # Pretrained model.
    EMBEDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"  # Pretrained model.
    EMBEDING_MODEL_PATH: str = "embeding_llm"  # Pretrained model path.
    DL_MAX_SEQ_LEN: int = 512  # Maximum sequence length.
    DL_EMBEDDING_ROOT: str = "data/embeddings"
    DL_EMBEDDING_DIR: str = str(Path(DL_EMBEDDING_ROOT) / "iece")
    DL_HIDDEN_SIZE: int = 1024  # PLM hidden size for Qwen3-Embedding-0.6B.

    # Custom feature extractor dimensions from best_hparams_imr_v1.json, trial #32.
    DL_D_MODEL: int = 320  # Transformer feature extractor d_model.
    DL_NHEAD: int = 8  # Number of attention heads: 320 / 8 = 40.
    DL_DIM_FF: int = 640  # Transformer feedforward dimension: 2x d_model.
    DL_N_EMOTION_LAYERS: int = 5  # Number of EmotionEncoder layers.
    DL_N_CAUSE_LAYERS: int = (
        2  # Number of CauseEncoder layers, including emotion prefix injection.
    )
    DL_DROPOUT: float = 0.25  # Dropout rate.

    # Training hyperparameters.
    DL_BATCH_SIZE: int = 64  # Batch size.
    DL_LEARNING_RATE: float = 9.081049133467078e-05  # Initial learning rate.
    DL_WEIGHT_DECAY: float = 0.0004040889483350774  # Weight decay.

    # Branch learning-rate multipliers: final lr = DL_LEARNING_RATE * multiplier.
    LR_SHARED: float = 1.0
    LR_EMO: float = 1.0
    LR_CAUSE: float = 1.0

    # LR schedule with ReduceLROnPlateau.
    DL_WARMUP_EPOCHS: int = 5  # Linear warmup epochs.
    DL_LR_PLATEAU_PATIENCE: int = 8  # Plateau patience before multiplying LR by 0.5.

    # Training epochs.
    DL_EPOCHS: int = 80  # Maximum supervised IECE epochs, controlled by early stopping.
    DL_EARLY_STOP_PATIENCE: int = 10  # Early-stopping patience.

    # Loss configuration.
    DL_LABEL_SMOOTHING: float = 0.05  # Label smoothing.
    LOSS_WEIGHT_EMOTION: float = 1.0
    LOSS_WEIGHT_CAUSE: float = 1.0

    # --- IMR (Iterative Mutual Refinement) architecture parameters ---
    # Ablation modes:
    # - "full": full bidirectional IMR
    # - "wo_imr": remove cross-task interaction; both branches are independent
    # - "wo_backward": remove cause-to-emotion feedback and keep emotion-to-cause only
    DL_ABLATION_MODE: str = "full"  # Ablation mode for controlling information flow.

    # Number of interaction rounds controlling emotion-cause refinement depth.
    DL_IMR_ITERATIONS: int = 4  # Iteration count T.

    # Whether to use event-specific embeddings for cause extraction.
    # True: use standalone event_emb, encoded from event text only.
    # False: use concat_emb, encoded from text and event together as in v2.
    DL_USE_EVENT_EMB: bool = True  # Enable task-aligned event-specific embeddings.

    @classmethod
    def apply_dataset_type(cls, dataset_type: str | None = None):
        """Keep the runtime fixed to the IECE paper-demo dataset."""
        dataset_type = (dataset_type or "IECE").upper().strip()
        if dataset_type != "IECE":
            raise ValueError("This demo only supports DATASET_TYPE='IECE'.")

        cls.DATASET_TYPE = dataset_type
        cls.DATASET = cls.IECE_DATASET
        cls.LABELS_MAP_E2C = cls.IECE_LABELS_MAP_E2C
        cls.NUM_EMOTION_CATEGORIES = len(cls.IECE_EMO_DICT)
        cls.DL_EMBEDDING_DIR = str(Path(cls.DL_EMBEDDING_ROOT) / "iece")

    @classmethod
    def get_emotion_to_idx(cls) -> dict:
        return cls.IECE_EMO_DICT

    @classmethod
    def get_num_emotions(cls) -> int:
        return cls.NUM_EMOTION_CATEGORIES

    @classmethod
    def get_lr_dict(cls, base_lr: float | None = None) -> dict:
        lr = cls.DL_LEARNING_RATE if base_lr is None else base_lr
        return {
            "shared": lr * cls.LR_SHARED,
            "emotion": lr * cls.LR_EMO,
            "cause": lr * cls.LR_CAUSE,
        }


# Initialize dynamic configuration.
Config.apply_dataset_type(Config.DATASET_TYPE)
