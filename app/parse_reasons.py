"""
Parse scorer reason strings into (label, value) pairs for charting.

Reason strings contain '(+N)' or '(-N)' (integer) somewhere in the string.
Examples:
    "Genre profile (+13): Drama, Crime"           → ("Genre profile", 13)
    "Drama + Crime combination (+11) — core..."   → ("Drama + Crime combination", 11)
    "High IMDb (7.5) on spectacle genre (-12)..."  → ("High IMDb (7.5) on spectacle genre", -12)
    "Psychological tag (+5)"                       → ("Psychological tag", 5)
    "'slapstick' tag (-8)"                         → ("'slapstick' tag", -8)
"""

import re

# Matches the first occurrence of (+N) or (-N) with an integer value.
# Uses non-greedy .*? so it captures up to the *first* signed integer in parens.
_RE = re.compile(r"^(.*?)\s*\(([+-]\d+)\)")


def parse_reason(s: str) -> tuple[str, int]:
    """Parse a single reason string into (label, value).

    Em-dash annotations (— ...) are stripped from the label.
    Returns (original_string, 0) if no signed integer in parens is found.
    """
    m = _RE.match(s)
    if not m:
        return s, 0
    # group(1) is always everything before the (+N) — the em-dash splitting
    # is intentionally omitted because labels like "8 episodes — compact commitment"
    # contain meaningful em-dashes that must not be truncated.
    return m.group(1).strip(), int(m.group(2))


def parse_all(reasons: list[str]) -> list[tuple[str, int]]:
    """Parse a list of reason strings, dropping zero-value entries."""
    parsed = [parse_reason(r) for r in reasons]
    return [(label, val) for label, val in parsed if val != 0]
