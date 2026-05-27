"""
IECE dataset utilities for the paper demo.

The demo trains only on event-level implicit emotion cause labels from
data/Implicit_emotion_cause_dataset.xml, while retaining the sample-level
emotion label as an auxiliary multi-task training target. Embeddings are
precomputed by src.prepare_embeddings and loaded lazily with numpy mmap.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from data.iece_data_loader import IECEDataLoader, IECESample
from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class EventPair:
    """Single event-level IECE training item."""

    text: str
    event_text: str
    emotion_label: int
    cause_label: int
    doc_id: str

    @classmethod
    def from_sample(cls, sample: IECESample) -> List["EventPair"]:
        emotion_label = Config.get_emotion_to_idx().get(sample.emotion_category, -1)
        pairs: List[EventPair] = []
        for event in sample.events:
            pairs.append(
                cls(
                    text=sample.original_text,
                    event_text=event.event_text,
                    emotion_label=emotion_label,
                    cause_label=1 if event.cause == "Y" else 0,
                    doc_id=str(sample.sample_id),
                )
            )
        return pairs


def flatten_samples(samples: List[IECESample]) -> List[EventPair]:
    """Flatten IECE samples into event-level pairs."""
    pairs: List[EventPair] = []
    for sample in samples:
        pairs.extend(EventPair.from_sample(sample))
    return pairs


def get_kfold_splits(
    labeled_samples: List[IECESample],
    n_splits: int = 10,
    random_state: int = Config.RANDOM,
) -> List[Tuple[List[int], List[int]]]:
    """Split at sample level to avoid leaking events from one text across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    emotion_labels = [str(s.emotion_category or "UNK") for s in labeled_samples]
    emotion_counter = Counter(emotion_labels)
    if len(emotion_counter) > 1 and min(emotion_counter.values()) >= n_splits:
        strat_labels = emotion_labels
        strat_name = "emotion_category"
    else:
        has_cause_labels = [int(len(s.cause_events) > 0) for s in labeled_samples]
        has_cause_counter = Counter(has_cause_labels)
        if len(has_cause_counter) > 1 and min(has_cause_counter.values()) >= n_splits:
            strat_labels = has_cause_labels
            strat_name = "has_cause"
        else:
            event_bucket_labels = [min(len(s.events) // 4, 5) for s in labeled_samples]
            event_bucket_counter = Counter(event_bucket_labels)
            if (
                len(event_bucket_counter) > 1
                and min(event_bucket_counter.values()) >= n_splits
            ):
                strat_labels = event_bucket_labels
                strat_name = "event_count_bucket"
            else:
                strat_labels = [i % n_splits for i in range(len(labeled_samples))]
                strat_name = "index_mod_fallback"
                logger.warning(
                    "Could not build stable real stratification labels; "
                    "falling back to index_mod stratification."
                )

    indices = np.arange(len(labeled_samples))
    splits = []
    for train_idx, val_idx in skf.split(indices, strat_labels):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    logger.info(f"K-fold split complete: {n_splits} folds | stratification={strat_name}")
    return splits


class IECEDataset(Dataset):
    """EventPair dataset backed by precomputed text/concat/event embeddings."""

    def __init__(
        self,
        pairs: List[EventPair],
        text_embs: np.ndarray,
        concat_embs: np.ndarray,
        event_embs: np.ndarray,
        indices: Optional[List[int]] = None,
        text_masks: Optional[np.ndarray] = None,
        event_masks: Optional[np.ndarray] = None,
    ):
        if indices is None:
            assert (
                len(pairs)
                == text_embs.shape[0]
                == concat_embs.shape[0]
                == event_embs.shape[0]
            ), (
                "Sample count mismatch: "
                f"pairs={len(pairs)}, text_embs={text_embs.shape[0]}, "
                f"concat_embs={concat_embs.shape[0]}, event_embs={event_embs.shape[0]}"
            )
        else:
            assert len(pairs) == len(indices), (
                f"pairs and indices count mismatch: pairs={len(pairs)}, "
                f"indices={len(indices)}"
            )

        self.pairs = pairs
        self.text_embs = text_embs
        self.concat_embs = concat_embs
        self.event_embs = event_embs
        self.indices = indices
        self.text_masks = text_masks
        self.event_masks = event_masks

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        emb_idx = self.indices[idx] if self.indices is not None else idx

        result = {
            "text_emb": torch.from_numpy(
                self.text_embs[emb_idx].astype(np.float32)
            ),
            "concat_emb": torch.from_numpy(
                self.concat_embs[emb_idx].astype(np.float32)
            ),
            "event_emb": torch.from_numpy(
                self.event_embs[emb_idx].astype(np.float32)
            ),
            "emotion_label": torch.tensor(pair.emotion_label, dtype=torch.long),
            "cause_label": torch.tensor(pair.cause_label, dtype=torch.long),
            "text": pair.text,
            "event_text": pair.event_text,
            "doc_id": pair.doc_id,
        }

        if self.text_masks is not None:
            result["key_padding_mask"] = torch.from_numpy(
                self.text_masks[emb_idx].astype(bool)
            )
        if self.event_masks is not None:
            result["event_padding_mask"] = torch.from_numpy(
                self.event_masks[emb_idx].astype(bool)
            )
        return result


def build_dataloaders(
    fold_train_pairs: List[EventPair],
    fold_val_pairs: List[EventPair],
    labeled_text_embs: np.ndarray,
    labeled_concat_embs: np.ndarray,
    labeled_event_embs: np.ndarray,
    fold_train_indices: List[int],
    fold_val_indices: List[int],
    labeled_text_masks: Optional[np.ndarray] = None,
    labeled_event_masks: Optional[np.ndarray] = None,
    batch_size: int = Config.DL_BATCH_SIZE,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and labeled train loaders for one fold."""
    train_dataset = IECEDataset(
        fold_train_pairs,
        labeled_text_embs,
        labeled_concat_embs,
        labeled_event_embs,
        indices=fold_train_indices,
        text_masks=labeled_text_masks,
        event_masks=labeled_event_masks,
    )
    val_dataset = IECEDataset(
        fold_val_pairs,
        labeled_text_embs,
        labeled_concat_embs,
        labeled_event_embs,
        indices=fold_val_indices,
        text_masks=labeled_text_masks,
        event_masks=labeled_event_masks,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"DataLoaders built: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches"
    )
    return train_loader, val_loader, train_loader


def load_all_data(
    batch_size: int = Config.DL_BATCH_SIZE,
) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
    """Load IECE embeddings and return 10-fold DataLoaders."""
    emb_dir = Path(Config.DL_EMBEDDING_DIR)
    if not (emb_dir / "labeled_text.npy").exists():
        raise FileNotFoundError(
            f"Precomputed embeddings were not found in {emb_dir}. "
            "Run: uv run python -m src.prepare_embeddings"
        )

    logger.info("Loading precomputed embeddings with lazy mmap...")
    labeled_text_embs = np.load(emb_dir / "labeled_text.npy", mmap_mode="r")
    labeled_concat_embs = np.load(emb_dir / "labeled_concat.npy", mmap_mode="r")
    labeled_event_embs = np.load(emb_dir / "labeled_event.npy", mmap_mode="r")
    logger.info(f"  labeled_text.npy: {labeled_text_embs.shape}")
    logger.info(f"  labeled_event.npy: {labeled_event_embs.shape}")

    labeled_text_masks: Optional[np.ndarray] = None
    labeled_event_masks: Optional[np.ndarray] = None
    if (emb_dir / "labeled_text_mask.npy").exists():
        labeled_text_masks = np.load(
            emb_dir / "labeled_text_mask.npy", mmap_mode="r"
        )
        logger.info(f"  labeled_text_mask.npy: {labeled_text_masks.shape} [loaded]")
    else:
        logger.warning("  labeled_text_mask.npy is missing; falling back to no-mask mode.")

    if (emb_dir / "labeled_event_mask.npy").exists():
        labeled_event_masks = np.load(
            emb_dir / "labeled_event_mask.npy", mmap_mode="r"
        )
        logger.info(f"  labeled_event_mask.npy: {labeled_event_masks.shape} [loaded]")
    else:
        logger.warning("  labeled_event_mask.npy is missing; falling back to no-mask mode.")

    labeled_loader_obj = IECEDataLoader(data_path=Path(Config.DATASET))
    labeled_samples = labeled_loader_obj.load_data()
    all_labeled_pairs = flatten_samples(labeled_samples)
    assert len(all_labeled_pairs) == labeled_text_embs.shape[0], (
        f"Embedding count mismatch: pairs={len(all_labeled_pairs)}, "
        f"embs={labeled_text_embs.shape[0]}"
    )

    sample_to_pair_indices: List[List[int]] = []
    pair_offset = 0
    for sample in labeled_samples:
        n_events = len(sample.events)
        sample_to_pair_indices.append(list(range(pair_offset, pair_offset + n_events)))
        pair_offset += n_events

    folds = []
    for fold_idx, (train_sample_idx, val_sample_idx) in enumerate(
        get_kfold_splits(labeled_samples)
    ):
        train_indices = [
            pi for si in train_sample_idx for pi in sample_to_pair_indices[si]
        ]
        val_indices = [pi for si in val_sample_idx for pi in sample_to_pair_indices[si]]

        train_pairs = [all_labeled_pairs[i] for i in train_indices]
        val_pairs = [all_labeled_pairs[i] for i in val_indices]
        logger.info(
            f"Fold {fold_idx + 1}/10: train={len(train_pairs)} pairs, "
            f"val={len(val_pairs)} pairs"
        )

        folds.append(
            build_dataloaders(
                fold_train_pairs=train_pairs,
                fold_val_pairs=val_pairs,
                labeled_text_embs=labeled_text_embs,
                labeled_concat_embs=labeled_concat_embs,
                labeled_event_embs=labeled_event_embs,
                fold_train_indices=train_indices,
                fold_val_indices=val_indices,
                labeled_text_masks=labeled_text_masks,
                labeled_event_masks=labeled_event_masks,
                batch_size=batch_size,
            )
        )
        logger.info(f"Fold {fold_idx + 1}/10 is ready")

    return folds
