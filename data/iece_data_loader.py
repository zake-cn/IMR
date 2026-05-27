"""
IECE data loading module.
Loads and processes Implicit Emotion Cause Extraction (IECE) data.
"""

import sys
from pathlib import Path

# Add the project root to the Python path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import xml.etree.ElementTree as ET
import json
import ast
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data structure."""

    event_id: int
    cause: str  # "Y" or "N"
    event_structure: Dict  # Structured event representation.

    @property
    def is_cause(self) -> bool:
        return self.cause == "Y"

    @property
    def event_text(self) -> str:
        """Extract the full event text from the structured event."""
        parts = []
        structure = self.event_structure

        # Assemble event text in the original field order.
        if att_subj := structure.get("Att_Subj"):
            parts.append(att_subj)
        if subj := structure.get("Subj"):
            parts.append(subj)
        if adv := structure.get("Adv"):
            parts.append(adv)
        if p := structure.get("P"):
            parts.append(p)
        if cpl := structure.get("Cpl"):
            parts.append(cpl)
        if att_obj := structure.get("Att_Obj"):
            parts.append(att_obj)
        if obj := structure.get("Obj"):
            parts.append(obj)

        # Drop empty fragments and concatenate the remaining parts.
        text = "".join(part for part in parts if part.strip())
        if text:
            return text
        if fallback_text := structure.get("text"):
            return str(fallback_text)
        return str(structure)


@dataclass
class IECESample:
    """IECE sample data structure."""

    sample_id: str
    emotion_category: str
    original_text: str
    events: List[Event]
    index: int = -1

    @property
    def cause_events(self) -> List[Event]:
        """Return events marked as causes."""
        return [event for event in self.events if event.is_cause]

    @property
    def non_cause_events(self) -> List[Event]:
        """Return events not marked as causes."""
        return [event for event in self.events if not event.is_cause]


class IECEDataLoader:
    """IECE data loader."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        data_range: Optional[tuple] = None,
    ):
        self.data_path = data_path or Config.DATASET
        self.data_range = data_range
        self.samples: List[IECESample] = []

    def load_data(self) -> List[IECESample]:
        """Load IECE data."""
        try:
            tree = ET.parse(self.data_path)
            root = tree.getroot()

            samples = []
            for sample_elem in root.findall("sample"):
                sample = self._parse_sample(sample_elem)
                if sample:
                    samples.append(sample)

            # Apply an optional data range.
            if self.data_range:
                start, end = self.data_range
                samples = samples[start:end]
                # Store indices relative to the original dataset.
                for i, sample in enumerate(samples):
                    sample.index = start + i
            else:
                # Store indices for the full dataset.
                for i, sample in enumerate(samples):
                    sample.index = i

            self.samples = samples
            logger.info(f"Loaded {len(self.samples)} IECE samples")
            return self.samples

        except Exception as e:
            logger.error(f"Failed to load IECE data: {e}")
            return []

    def _parse_sample(self, sample_elem) -> Optional[IECESample]:
        """Parse one sample element."""
        try:
            sample_id = sample_elem.get("id")
            emotion_category = (
                sample_elem.get("emotion_category")
                or sample_elem.get("emotion")
                or sample_elem.get("label")
            )

            # Read the original text.
            original_text_elem = sample_elem.find("original_text")
            original_text = (
                original_text_elem.text.strip()
                if original_text_elem is not None
                else ""
            )

            # Parse event elements.
            events = []
            events_elem = sample_elem.find("events")
            if events_elem is not None:
                for event_elem in events_elem.findall("event"):
                    event = self._parse_event(event_elem)
                    if event:
                        events.append(event)

            return IECESample(
                sample_id=sample_id,
                emotion_category=emotion_category,
                original_text=original_text,
                events=events,
            )

        except Exception as e:
            logger.warning(f"Failed to parse sample: {e}")
            return None

    def _parse_event(self, event_elem) -> Optional[Event]:
        """Parse one event element."""
        try:
            event_id_raw = event_elem.get("id")
            event_id = int(event_id_raw) if event_id_raw is not None else -1
            cause = (event_elem.get("cause") or "N").upper()

            # IECE event text is usually stored in event_elem.text as a structured dict.
            event_text_elem = event_elem.find("event_text")
            if event_text_elem is not None and event_text_elem.text:
                event_text = event_text_elem.text.strip()
            else:
                event_text = event_elem.text.strip() if event_elem.text else ""

            try:
                event_structure = ast.literal_eval(event_text)
                if not isinstance(event_structure, dict):
                    event_structure = {"text": event_text}
            except (ValueError, SyntaxError):
                event_structure = {"text": event_text}

            return Event(
                event_id=event_id,
                cause=cause,
                event_structure=event_structure,
            )

        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None

    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        if not self.samples:
            return {}

        total_samples = len(self.samples)
        total_events = sum(len(sample.events) for sample in self.samples)
        total_cause_events = sum(len(sample.cause_events) for sample in self.samples)
        total_non_cause_events = sum(
            len(sample.non_cause_events) for sample in self.samples
        )

        # Count the emotion category distribution.
        emotion_categories = {}
        for sample in self.samples:
            emotion = sample.emotion_category
            emotion_categories[emotion] = emotion_categories.get(emotion, 0) + 1

        return {
            "total_samples": total_samples,
            "total_events": total_events,
            "cause_events": total_cause_events,
            "non_cause_events": total_non_cause_events,
            "avg_events_per_sample": (
                total_events / total_samples if total_samples > 0 else 0
            ),
            "avg_cause_events_per_sample": (
                total_cause_events / total_samples if total_samples > 0 else 0
            ),
            "emotion_category_distribution": emotion_categories,
        }

    def get_sample_by_id(self, sample_id: str) -> Optional[IECESample]:
        """Return a sample by ID."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = IECEDataLoader()
    samples = loader.load_data()
    stats = loader.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(samples[0])
