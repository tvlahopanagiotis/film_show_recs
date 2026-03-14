"""
SessionPrefs dataclass for TV show recommendations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionPrefs:
    era_min_year: Optional[int] = None       # None = no filter
    status_filter: str = "any"               # "ended", "ongoing", "any"
    max_episodes: Optional[int] = None       # total aired episodes; None = no filter
    exclude_famous: bool = False             # exclude shows above famous_votes_threshold
    famous_votes_threshold: int = 200_000    # Trakt vote count threshold
    mood: str = "any"                        # "tense", "thoughtful", "lighter", "any"
