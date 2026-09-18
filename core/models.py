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


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list = field(default_factory=list)


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
