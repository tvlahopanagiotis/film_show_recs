"""
SessionPrefs dataclass — holds per-run user preferences from the interactive prompt.
Imported by both recommend.py and candidates.py to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionPrefs:
    era_min_year: Optional[int] = None      # None = no filter
    era_max_year: Optional[int] = None      # None = no filter
    exclude_famous: bool = False            # exclude films above famous_votes_threshold
    famous_votes_threshold: int = 500_000  # what counts as "famous" (TMDB vote count)
    mood: str = "any"                      # "tense", "thoughtful", "lighter", "any"
    max_runtime: Optional[int] = None      # minutes; None = no filter
