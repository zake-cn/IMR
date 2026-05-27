"""
Offline embedding precomputation script.

Uses the PLM as a static feature extractor to generate offline sequence
embeddings for all EventPair instances, then saves them as float16 numpy files.

Training only needs to load the .npy files, so the PLM does not need to run
again and training is much faster.

Usage:
    python src/prepare_embeddings.py

Output directory: Config.DL_EMBEDDING_DIR, default data/embeddings/.
    labeled_text.npy        shape (N_labeled, L, H) float16
    labeled_concat.npy      shape (N_labeled, L, H) float16
    augmented_text.npy      shape (N_aug, L, H)     float16
    augmented_concat.npy    shape (N_aug, L, H)     float16
    meta.json               records shapes, model name, and metadata
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import Optional
from modelscope import snapshot_download

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.config import Config
from data.iece_data_loader import IECEDataLoader
from src.dataset import EventPair, flatten_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core extraction function.
# ---------------------------------------------------------------------------


def extract_embeddings_to_memmap(
    pairs: list[EventPair],
    tokenizer,
    model,
    device: torch.device,
    text_path: Path,
    concat_path: Path,
    event_path: Path,  # Event-only embedding.
    text_mask_path: Optional[Path] = None,
    concat_mask_path: Optional[Path] = None,
    event_mask_path: Optional[Path] = None,  # Event mask.
    max_length: int = Config.DL_MAX_SEQ_LEN,
    batch_size: int = 32,
) -> tuple[int, int]:
    """
    Encode each (text, event_text) pair and extract full sequence embeddings.

    Generates three embedding types:
      1. text_emb: plain text, used by the IMR text-state branch
      2. concat_emb: text and event encoded together
      3. event_emb: event-only text, used by the IMR cause branch

    Writes each batch directly to disk with np.memmap, so RAM only holds one
    batch at a time.

    When text_mask_path / concat_mask_path / event_mask_path is not None, it
    also stores attention masks using True for padding positions, matching the
    PyTorch src_key_padding_mask convention.

    Returns:
        N: number of samples
        H: hidden_size
    """
    N = len(pairs)
    H = model.config.hidden_size
    L = max_length

    # Create memmap files with standard .npy headers for np.load(mmap_mode='r').
    text_embs = np.lib.format.open_memmap(
        text_path, mode="w+", dtype=np.float16, shape=(N, L, H)
    )
    concat_embs = np.lib.format.open_memmap(
        concat_path, mode="w+", dtype=np.float16, shape=(N, L, H)
    )
    event_embs = np.lib.format.open_memmap(
        event_path, mode="w+", dtype=np.float16, shape=(N, L, H)
    )

    # Optionally create attention mask files, where True means padding.
    text_masks_mm = (
        np.lib.format.open_memmap(
            text_mask_path, mode="w+", dtype=np.bool_, shape=(N, L)
        )
        if text_mask_path is not None
        else None
    )
    concat_masks_mm = (
        np.lib.format.open_memmap(
            concat_mask_path, mode="w+", dtype=np.bool_, shape=(N, L)
        )
        if concat_mask_path is not None
        else None
    )
    event_masks_mm = (
        np.lib.format.open_memmap(
            event_mask_path, mode="w+", dtype=np.bool_, shape=(N, L)
        )
        if event_mask_path is not None
        else None
    )

    model.eval()
    with torch.no_grad():
        for step, start in enumerate(
            tqdm(range(0, N, batch_size), desc="Extracting batches")
        ):
            batch = pairs[start : start + batch_size]
            bs = len(batch)

            texts = [p.text for p in batch]
            events = [p.event_text for p in batch]

            # 1. Plain text encoding for the IMR text-state branch.
            text_enc = tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # 2. Joint text+event encoding, usable by the cause branch.
            concat_enc = tokenizer(
                texts,
                events,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # 3. Event-only encoding for the IMR cause extraction branch.
            event_enc = tokenizer(
                events,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            text_out = model(**text_enc).last_hidden_state  # (bs, L, H)
            concat_out = model(**concat_enc).last_hidden_state  # (bs, L, H)
            event_out = model(**event_enc).last_hidden_state  # (bs, L, H)

            text_embs[start : start + bs] = (
                text_out.cpu().float().numpy().astype(np.float16)
            )
            concat_embs[start : start + bs] = (
                concat_out.cpu().float().numpy().astype(np.float16)
            )
            event_embs[start : start + bs] = (
                event_out.cpu().float().numpy().astype(np.float16)
            )

            # attention_mask uses 1=real token and 0=padding; convert to True=padding.
            if text_masks_mm is not None:
                text_masks_mm[start : start + bs] = (
                    text_enc["attention_mask"].cpu().numpy() == 0
                )
            if concat_masks_mm is not None:
                concat_masks_mm[start : start + bs] = (
                    concat_enc["attention_mask"].cpu().numpy() == 0
                )
            if event_masks_mm is not None:
                event_masks_mm[start : start + bs] = (
                    event_enc["attention_mask"].cpu().numpy() == 0
                )

            # Flush every 100 batches to avoid accumulating dirty OS page cache.
            if step % 100 == 0:
                text_embs.flush()
                concat_embs.flush()
                event_embs.flush()
                if text_masks_mm is not None:
                    text_masks_mm.flush()
                if concat_masks_mm is not None:
                    concat_masks_mm.flush()
                if event_masks_mm is not None:
                    event_masks_mm.flush()

    text_embs.flush()
    concat_embs.flush()
    event_embs.flush()
    if text_masks_mm is not None:
        text_masks_mm.flush()
        del text_masks_mm
    if concat_masks_mm is not None:
        concat_masks_mm.flush()
        del concat_masks_mm
    if event_masks_mm is not None:
        event_masks_mm.flush()
        del event_masks_mm
    del text_embs, concat_embs, event_embs

    return N, H


# ---------------------------------------------------------------------------
# Main flow.
# ---------------------------------------------------------------------------


def pre_embed(dataset_type: Optional[str] = None):
    if dataset_type is not None:
        Config.apply_dataset_type(dataset_type)

    device = Config.DEVICE
    emb_dir = Path(Config.DL_EMBEDDING_DIR)
    emb_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Dataset type: {Config.DATASET_TYPE}")
    logger.info(f"Data path: {Config.DATASET}")
    logger.info(f"Embedding output directory: {emb_dir}")
    logger.info(f"Loading PLM: {Config.EMBEDING_MODEL_PATH}")

    if Path(Config.EMBEDING_MODEL_PATH).exists() and any(
        Path(Config.EMBEDING_MODEL_PATH).iterdir()
    ):
        pass
    else:
        Path(Config.EMBEDING_MODEL_PATH).mkdir(parents=True, exist_ok=True)
        snapshot_download(
            model_id=Config.EMBEDING_MODEL,
            local_dir=str(Config.EMBEDING_MODEL_PATH),
            revision="master",
        )

    tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDING_MODEL_PATH)
    plm = AutoModel.from_pretrained(Config.EMBEDING_MODEL_PATH)
    plm = plm.to(device)
    plm.eval()
    for param in plm.parameters():
        param.requires_grad = False

    hidden_size: int = plm.config.hidden_size
    logger.info(f"PLM hidden_size = {hidden_size}")

    # ----------------------------------------------------------------
    # Process labeled IECE data from Implicit_emotion_cause_dataset.xml.
    # ----------------------------------------------------------------
    logger.info(f"Loading labeled data: {Config.DATASET}")
    labeled_dl = IECEDataLoader(data_path=Path(Config.DATASET))
    labeled_samples = labeled_dl.load_data()
    labeled_pairs = flatten_samples(labeled_samples)
    logger.info(f"  Labeled samples: {len(labeled_samples)} -> {len(labeled_pairs)} pairs")

    n_labeled, _ = extract_embeddings_to_memmap(
        labeled_pairs,
        tokenizer,
        plm,
        device,
        text_path=emb_dir / "labeled_text.npy",
        concat_path=emb_dir / "labeled_concat.npy",
        event_path=emb_dir / "labeled_event.npy",
        text_mask_path=emb_dir / "labeled_text_mask.npy",
        concat_mask_path=emb_dir / "labeled_concat_mask.npy",
        event_mask_path=emb_dir / "labeled_event_mask.npy",
    )
    labeled_size_mb = (
        n_labeled
        * Config.DL_MAX_SEQ_LEN
        * hidden_size
        * 2
        * 3
        / 1024
        / 1024  # Three embedding types.
    )
    logger.info(
        f"  Saved labeled_text.npy + labeled_concat.npy + labeled_event.npy "
        f"shape=({n_labeled}, {Config.DL_MAX_SEQ_LEN}, {hidden_size}) "
        f"approximately {labeled_size_mb:.0f} MB"
    )

    # ----------------------------------------------------------------
    # Release PLM memory after all embeddings have been written to disk.
    # ----------------------------------------------------------------
    del plm, tokenizer
    torch.cuda.empty_cache()
    logger.info("Embedding model memory released")

    # ----------------------------------------------------------------
    # Metadata.
    # ----------------------------------------------------------------
    meta = {
        "pretrained_model": Config.EMBEDING_MODEL,
        "seq_len": Config.DL_MAX_SEQ_LEN,
        "hidden_size": hidden_size,
        "n_labeled_pairs": len(labeled_pairs),
        "dtype": "float16",
    }
    with open(emb_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Metadata written to {emb_dir / 'meta.json'}")
    logger.info("Embedding precomputation complete.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Offline embedding precomputation")
    parser.parse_args()

    pre_embed()


if __name__ == "__main__":
    main()
