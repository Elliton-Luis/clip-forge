"""models.py — entidades de domínio. Zero dependência externa.

SOLID: SRP (só dados). Candidate concentra regra de overlap (coesão).
"""
from dataclasses import dataclass, field
from .config import OVERLAP_REJECT_RATIO


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None
    # Origem do timestamp: "whisper" (default, caminho produtivo inalterado)
    # ou "forced_alignment" (só quando o aligner re-mediu a palavra).
    timestamp_source: str = "whisper"


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list = field(default_factory=list)
    timestamp_source: str = "whisper"


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    words: list
    score: float = 0.0
    reason: str = ""
    title: str = ""
    hashtags: str = ""
    failed: bool = False
    energy: str = "media"
    speech_rate: float = 0.0
    original_start: float = 0.0
    original_end: float = 0.0
    snapped: bool = False
    # --- Peak (modo experimental --selection-mode peak; classic ignora) ---
    window_start: float | None = None  # bounds da janela original (pré-peak)
    window_end: float | None = None
    peak_start: float | None = None  # trecho de maior interesse
    peak_end: float | None = None
    peak_score: float = 0.0  # intensidade do auge (0..10)
    peak_source: str = "none"  # none|heuristic|llm
    peak_reason: str = ""
    title_source: str = "window"  # window|peak

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap_ratio(self, other: "Candidate") -> float:
        latest = max(self.start, other.start)
        earliest = min(self.end, other.end)
        overlap = max(0.0, earliest - latest)
        shortest = min(self.duration, other.duration)
        return (overlap / shortest) if shortest > 0 else 0.0

    def overlaps(self, other: "Candidate", ratio: float = OVERLAP_REJECT_RATIO) -> bool:
        return self.overlap_ratio(other) > ratio
