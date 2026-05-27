from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from src.config import Config

# ---------------------------------------------------------------------------
# Attention pooling.
# ---------------------------------------------------------------------------


class AttentionPool(nn.Module):
    """
    Attention pooling with learnable weights.

    This lets the model focus on the most relevant tokens in the sequence and
    gives cause-event representations stronger semantic localization than
    fixed mean pooling.

    scores = Linear(hidden) -> softmax with an optional padding mask
    output = Σ scores_t * hidden_t
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_weight = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden: torch.Tensor,  # (B, L, d_model)
        mask: Optional[torch.Tensor],  # (B, L) True=padding; may be None.
    ) -> torch.Tensor:  # (B, d_model)
        scores = self.attn_weight(hidden).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, L)
        # Guard against all-padding rows, which would otherwise produce NaNs.
        weights = torch.nan_to_num(weights, nan=0.0)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)


# ---------------------------------------------------------------------------
# Text encoder.
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """
    Text encoder for emotion features.

    Encodes token sequences with N Transformer layers to extract contextual
    features for implicit emotion recognition.

    Uses PyTorch's built-in TransformerEncoderLayer with norm_first=True
    (Pre-LN) for more stable training and better deep-gradient flow.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
    ) -> torch.Tensor:  # (B, L, d_model)
        return self.transformer(x, src_key_padding_mask=key_padding_mask)


# ---------------------------------------------------------------------------
# Loss.
# ---------------------------------------------------------------------------


class IMRLoss(nn.Module):
    """
    Multi-task training loss for the IECE demo.

    The main reported task is implicit emotion cause extraction, but the model
    still uses emotion recognition as an auxiliary training objective.
    """

    def __init__(
        self,
        label_smoothing: float,
        emotion_weight: float = Config.LOSS_WEIGHT_EMOTION,
        cause_weight: float = Config.LOSS_WEIGHT_CAUSE,
    ):
        super().__init__()
        self.emotion_ce = nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=label_smoothing,
        )
        self.cause_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.emotion_weight = float(emotion_weight)
        self.cause_weight = float(cause_weight)

    def forward(
        self,
        emotion_logits: torch.Tensor,
        cause_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        cause_labels: torch.Tensor,
    ):
        emotion_loss = self.emotion_ce(emotion_logits, emotion_labels)
        cause_loss = self.cause_ce(cause_logits, cause_labels)
        return self.emotion_weight * emotion_loss + self.cause_weight * cause_loss


# ===========================================================================
# IMR (Iterative Mutual Refinement) architecture v3.
# ===========================================================================
"""
IMR architecture for iterative emotion-cause joint reasoning.

Core ideas:
  1. **Task decoupling**: emotion and cause branches encode their own features.
  2. **Iterative interaction**: prediction runs T rounds of state refinement.
  3. **Event-specific embeddings**: the cause branch uses event_emb directly.

Inference flow:

  Inputs:
    text_emb:  (B, L_text, H)   # plain text for the text-state branch
    cause_input: (B, L_cause, H) # event_emb or concat_emb

  Stage 1 - independent feature encoding:
    emotion_feats = TextEncoder(text_emb)         (B, L_text, d)
    cause_feats   = TextEncoder(cause_input)        (B, L_cause, d)

  Stage 2 - initial state pooling:
    emo_state_0   = AttentionPool(emotion_feats)     (B, d)
    cause_state_0 = AttentionPool(cause_feats)       (B, d)

  Stage 3 - T rounds of iterative interaction:
    for t in range(T):
      # 1. Emotion update: gather clues from Cause
      emo_ctx = CrossAttentionInteraction(
                  query=emo_state_t, kv=cause_feats, kv_state=cause_state_t)
      emo_state_{t+1} = StateRefinement(emo_state_t, emo_ctx)

      # 2. Cause update: gather emotion priors from Emotion
      cause_ctx = CrossAttentionInteraction(
                    query=cause_state_t, kv=emotion_feats, kv_state=emo_state_{t+1})
      cause_state_{t+1} = StateRefinement(cause_state_t, cause_ctx)

  Stage 4 - final prediction:
    emotion_logits = EmotionHead(emo_state_T)        (B, 7)
    cause_logits   = CauseHead(cause_state_T)        (B, 2)

Benefits:
  - Two-branch design: text state and event-cause state reinforce each other.
  - Multi-round reasoning: repeatedly weighs emotion and cause relations.
  - Flexible control: interaction depth is controlled by iteration count T.
  - Task alignment: event_emb directly serves cause extraction.
"""


class CrossAttentionInteraction(nn.Module):
    """
    Cross-task interaction module.

    Extracts context related to the current task state from the other task's
    sequence features.

    query: (B, d) current task state vector
    key/value: (B, L, d) other task sequence features plus broadcast state
    output: (B, d) extracted interaction context

    Uses a small number of attention heads to reduce overfitting on small data.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,  # Use fewer heads to focus on global interaction.
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_state: torch.Tensor,  # (B, d) current task state
        kv_sequence: torch.Tensor,  # (B, L, d) other task sequence features
        kv_state: torch.Tensor,  # (B, d) other task state for key/value enhancement
        kv_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
    ) -> torch.Tensor:  # (B, d)
        # Broadcast the other task state into the sequence as a global signal.
        # kv_state: (B, d) -> (B, 1, d) -> broadcast to (B, L, d)
        kv_state_expanded = kv_state.unsqueeze(1).expand_as(kv_sequence)
        enhanced_kv = kv_sequence + kv_state_expanded  # (B, L, d) residual enhancement

        # query_state: (B, d) -> (B, 1, d) for attention
        q = query_state.unsqueeze(1)  # (B, 1, d)

        # The query extracts relevant information from the enhanced sequence.
        attn_out, _ = self.attn(
            query=q,
            key=enhanced_kv,
            value=enhanced_kv,
            key_padding_mask=kv_padding_mask,
        )  # (B, 1, d)
        return self.norm(attn_out.squeeze(1))  # (B, d)


class StateRefinementModule(nn.Module):
    """
    State update module.

    Fuses the previous state with the newly acquired cross-task context and
    returns the refined state.

    state_new = LayerNorm(state_old + FFN([state_old || cross_context]))

    Uses direct FFN fusion with a residual connection for stable gradients.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        state_old: torch.Tensor,  # (B, d)
        cross_context: torch.Tensor,  # (B, d) context extracted from the other task
    ) -> torch.Tensor:  # (B, d)
        # Concatenate the old state and cross-task context.
        fused_input = torch.cat([state_old, cross_context], dim=-1)  # (B, 2d)
        update = self.fusion(fused_input)  # (B, d)
        # Residual plus LayerNorm.
        return self.norm(state_old + update)


class IMRModel(nn.Module):
    """
    IMR (Iterative Mutual Refinement) model v3.

    Iterative joint reasoning architecture for emotion and cause.

    Args:
        hidden_size         PLM embedding dimension, default 1024.
        d_model             Transformer feature dimension.
        nhead               Number of attention heads.
        dim_feedforward     FFN hidden dimension.
        n_emotion_layers    Number of emotion encoder layers.
        n_cause_layers      Number of cause encoder layers.
        num_emotions        Number of emotion classes.
        n_iterations        Number of iterative interaction rounds T, default 3.
        dropout             Dropout rate.
        use_event_emb       Whether to use event_emb (True) or concat_emb (False).
        ablation_mode       Ablation mode: "full" | "wo_imr" | "wo_backward".
    """

    def __init__(
        self,
        hidden_size: int = Config.DL_HIDDEN_SIZE,
        d_model: int = Config.DL_D_MODEL,
        nhead: int = Config.DL_NHEAD,
        dim_feedforward: int = Config.DL_DIM_FF,
        n_emotion_layers: int = Config.DL_N_EMOTION_LAYERS,
        n_cause_layers: int = Config.DL_N_CAUSE_LAYERS,
        num_emotions: int = Config.get_num_emotions(),
        n_iterations: int = Config.DL_IMR_ITERATIONS,
        dropout: float = Config.DL_DROPOUT,
        use_event_emb: bool = Config.DL_USE_EVENT_EMB,
        ablation_mode: str = Config.DL_ABLATION_MODE,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_emotions = num_emotions
        self.n_iterations = n_iterations
        self.use_event_emb = use_event_emb
        valid_modes = {"full", "wo_imr", "wo_backward"}
        if ablation_mode not in valid_modes:
            raise ValueError(
                f"Invalid ablation_mode='{ablation_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )
        self.ablation_mode = ablation_mode

        # Input projections.
        self.emotion_proj = nn.Linear(hidden_size, d_model)
        self.cause_proj = nn.Linear(hidden_size, d_model)

        # ===== Stage 1: independent feature encoding =====
        self.emotion_encoder = TextEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_emotion_layers,
            dropout=dropout,
        )
        self.cause_encoder = TextEncoder(  # Same structure, independent parameters.
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_cause_layers,
            dropout=dropout,
        )

        # ===== Stage 2: initial state pooling =====
        self.emotion_attn_pool = AttentionPool(d_model)
        self.cause_attn_pool = AttentionPool(d_model)

        # ===== Stage 3: iterative interaction modules =====
        # Emotion update: gather clues from Cause.
        self.emo_from_cause_attn = CrossAttentionInteraction(d_model, dropout)
        self.emo_refinement = StateRefinementModule(d_model, dropout)

        # Cause update: gather emotion priors from Emotion.
        self.cause_from_emo_attn = CrossAttentionInteraction(d_model, dropout)
        self.cause_refinement = StateRefinementModule(d_model, dropout)

        # ===== Stage 4: final prediction heads =====
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_emotions),
        )

        self.cause_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(
        self,
        text_emb: torch.Tensor,  # (B, L, H) text embedding
        cause_input_emb: torch.Tensor,  # (B, L, H) event_emb or concat_emb
        text_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
        cause_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
        return_intermediate: bool = False,
    ):
        """
        Returns:
            emotion_logits: (B, num_emotions)  emotion class scores
            cause_logits:   (B, 2)             cause/non-cause scores
        """
        # ===== Stage 1: independent feature encoding =====
        emotion_seq = self.emotion_proj(text_emb)  # (B, L, d)
        emotion_feats = self.emotion_encoder(
            emotion_seq, text_padding_mask
        )  # (B, L, d)

        cause_seq = self.cause_proj(cause_input_emb)  # (B, L, d)
        cause_feats = self.cause_encoder(cause_seq, cause_padding_mask)  # (B, L, d)

        # ===== Stage 2: initial states =====
        emo_state = self.emotion_attn_pool(emotion_feats, text_padding_mask)  # (B, d)
        cause_state = self.cause_attn_pool(cause_feats, cause_padding_mask)  # (B, d)

        intermediates: List[Dict[str, torch.Tensor]] = []

        def _record(iteration_idx: int, emo: torch.Tensor, cau: torch.Tensor) -> None:
            if not return_intermediate:
                return
            intermediates.append(
                {
                    "iteration": torch.tensor(iteration_idx),
                    "emotion_logits": self.emotion_head(emo).detach().cpu(),
                    "cause_logits": self.cause_head(cau).detach().cpu(),
                }
            )

        # Variant 1: w/o IMR - both tasks are independent, with no interaction.
        if self.ablation_mode == "wo_imr":
            emotion_logits = self.emotion_head(emo_state)  # (B, num_emotions)
            cause_logits = self.cause_head(cause_state)  # (B, 2)
            _record(0, emo_state, cause_state)
            if return_intermediate:
                return emotion_logits, cause_logits, intermediates
            return emotion_logits, cause_logits

        # Variant 2: w/o Backward - keep only one-way Emotion -> Cause interaction.
        if self.ablation_mode == "wo_backward":
            for i in range(self.n_iterations):
                cause_ctx = self.cause_from_emo_attn(
                    query_state=cause_state,
                    kv_sequence=emotion_feats,
                    kv_state=emo_state,  # Emotion state remains the initial pooled state.
                    kv_padding_mask=text_padding_mask,
                )  # (B, d)
                cause_state = self.cause_refinement(cause_state, cause_ctx)  # (B, d)
                _record(i + 1, emo_state, cause_state)

            emotion_logits = self.emotion_head(emo_state)  # (B, num_emotions)
            cause_logits = self.cause_head(cause_state)  # (B, 2)
            if return_intermediate:
                return emotion_logits, cause_logits, intermediates
            return emotion_logits, cause_logits

        # ===== Stage 3: T rounds of iterative interaction =====
        for i in range(self.n_iterations):
            # 1. Emotion update: gather clues from Cause.
            emo_ctx = self.emo_from_cause_attn(
                query_state=emo_state,
                kv_sequence=cause_feats,
                kv_state=cause_state,
                kv_padding_mask=cause_padding_mask,
            )  # (B, d)
            emo_state = self.emo_refinement(emo_state, emo_ctx)  # (B, d)

            # 2. Cause update: gather emotion priors from Emotion.
            cause_ctx = self.cause_from_emo_attn(
                query_state=cause_state,
                kv_sequence=emotion_feats,
                kv_state=emo_state,  # Use the just-updated emotion state.
                kv_padding_mask=text_padding_mask,
            )  # (B, d)
            cause_state = self.cause_refinement(cause_state, cause_ctx)  # (B, d)
            _record(i + 1, emo_state, cause_state)

        # ===== Stage 4: final prediction =====
        emotion_logits = self.emotion_head(emo_state)  # (B, 7)
        cause_logits = self.cause_head(cause_state)  # (B, 2)

        if return_intermediate:
            return emotion_logits, cause_logits, intermediates
        return emotion_logits, cause_logits
