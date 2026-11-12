"""Diagnose which historical order-log lines can be safely re-partitioned.

Context: order-log lines contain several numbers, and parse(fuzzy=True) may
consume part of an order id as the date. Three evidence tiers are compared:

  A. the raw log line is retained
  B. only the parsed datetime + skipped tokens are retained (fuzzy_with_tokens)
  C. only the UTC partition (a date) is retained

Run: .venv/bin/python repartition_diagnosis.py
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from dateutil.parser import ParserError, parse

BUSINESS_TZ = ZoneInfo("Asia/Shanghai")

# Source-convention evidence: naive timestamps may come from any of these.
NAIVE_SOURCE_TZS = [timezone.utc, BUSINESS_TZ, ZoneInfo("America/New_York")]

# Evidence from log-file names/mtime: all historical files cover 2025.
DEFAULT_YEAR_CANDIDATES = [2025]

# ---------------------------------------------------------------------------
# Finite template family (source conventions for the date slot of a line).
# Order ids are "order#YYYYMMDD" and are NOT dates: the compact template
# refuses digits preceded by '#'.
# ---------------------------------------------------------------------------
TIME_RE = re.compile(
    r"(?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?"
    r"(?:\s*(?P<off>Z|[+-]\d{2}:?\d{2}))?"
)

TEMPLATES = [
    ("iso", re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"),
     ("y", "m", "d"), True),
    ("us", re.compile(r"(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4})"),
     ("y", "m", "d"), True),
    ("eu", re.compile(r"(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})"),
     ("y", "m", "d"), True),
    ("compact", re.compile(r"(?<![#\d])(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})(?!\d)"),
     ("y", "m", "d"), True),
    ("month_day", re.compile(r"(?<![\d-])(?P<m>\d{1,2})-(?P<d>\d{1,2})(?![\d-])"),
     ("y", "m", "d"), False),
]


def _parse_offset(off: str | None):
    if off is None:
        return None
    if off == "Z":
        return timezone.utc
    sign = 1 if off[0] == "+" else -1
    digits = off[1:].replace(":", "")
    from datetime import timedelta
    return timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))


def compatible_readings(line: str):
    """All (template, aware datetime, assumption) readings the family allows."""
    time_match = TIME_RE.search(line)
    hour = minute = second = 0
    offset = None
    if time_match:
        hour = int(time_match.group("H"))
        minute = int(time_match.group("M"))
        second = int(time_match.group("S") or 0)
        offset = _parse_offset(time_match.group("off"))

    readings = []
    for name, pattern, fields, has_year in TEMPLATES:
        for match in pattern.finditer(line):
            parts = {k: int(v) for k, v in match.groupdict().items()}
            years = [parts["y"]] if has_year else DEFAULT_YEAR_CANDIDATES
            for year in years:
                try:
                    naive = datetime(year, parts["m"], parts["d"],
                                     hour, minute, second)
                except ValueError:
                    continue  # e.g. month=13 under the "us" reading
                assumptions = []
                if not has_year:
                    assumptions.append(f"default-year={year}")
                tzs = [offset] if offset is not None else NAIVE_SOURCE_TZS
                if offset is None:
                    assumptions.append("assumed-tz")
                for tz in tzs:
                    readings.append((name, naive.replace(tzinfo=tz),
                                     tuple(assumptions)))
    return readings


def business_dates(line: str):
    """Map each compatible reading to its business date."""
    return [(name, dt.astimezone(BUSINESS_TZ).date(), assumptions)
            for name, dt, assumptions in compatible_readings(line)]


def classify(line: str):
    readings = business_dates(line)
    if not readings:
        return "QUARANTINE", "no template in the family matches", []
    dates = {d for _, d, _ in readings}
    assumed = sorted({a for _, _, asm in readings for a in asm})
    if len(dates) > 1:
        return "QUARANTINE", f"compatible business dates differ: {sorted(dates)}", readings
    if len(readings) > 1 or assumed:
        return "PASS", (f"single business day {dates.pop()} despite uncertainty "
                        f"({', '.join(assumed) or 'multiple templates'})"), readings
    return "SAFE", f"exactly one reading: {readings[0][0]} -> {dates.pop()}", readings


# ---------------------------------------------------------------------------
# Evidence tiers
# ---------------------------------------------------------------------------
def tier_b_observation(line: str):
    """What remains if only the fuzzy parse result + skipped tokens are kept."""
    try:
        dt, skipped = parse(line, fuzzy_with_tokens=True)
        return dt, skipped
    except ParserError as exc:
        return None, f"ParserError: {exc}"


def tier_c_observation(line: str):
    """What remains if only the UTC partition is kept."""
    dt, _ = tier_b_observation(line)
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).date()


HISTORICAL_LINES = [
    "2025-03-04 10:00:00+00:00 confirmed",
    "order#20231115 shipped 2025-03-04 10:00:00",
    "03/04/2025 10:00:00 confirmed",
    "13/04/2025 10:00:00 confirmed",
    "2025-03-04 23:30:00 confirmed",
    "2025-03-04 10:00:00 confirmed",
    "03-04 10:00 confirmed",
    "order#20231115 confirmed",
    "settlement 20250304 batch closed",
]

ADVERSARIAL_A = "2025-03-04 10:00:00 confirmed"   # source writes ISO
ADVERSARIAL_B = "03/04/2025 10:00:00 confirmed"   # source writes DD/MM/YYYY


def main():
    print("=" * 78)
    print("1) Naive fuzzy parse vs. template-family diagnosis (tier A: raw line kept)")
    print("=" * 78)
    for line in HISTORICAL_LINES:
        dt, skipped = tier_b_observation(line)
        fuzzy = dt.isoformat() if dt else skipped
        verdict, reason, readings = classify(line)
        print(f"\nline : {line!r}")
        print(f"fuzzy: {fuzzy}")
        print(f"family: compatible business dates = "
              f"{sorted({str(d) for _, d, _ in readings}) or 'none'}")
        for name, d, asm in readings:
            print(f"        - {name:<9} -> {d}  {','.join(asm)}")
        print(f"verdict: {verdict} -- {reason}")

    print()
    print("=" * 78)
    print("2) Adversarial pair: identical tier-B and tier-C evidence, different truth")
    print("=" * 78)
    for label, line, truth in [
        ("A (ISO source)", ADVERSARIAL_A, date(2025, 3, 4)),
        ("B (DD/MM source)", ADVERSARIAL_B, date(2025, 4, 3)),
    ]:
        dt, skipped = tier_b_observation(line)
        print(f"{label}: {line!r}")
        print(f"   tier B: parsed={dt.isoformat()} skipped={skipped}")
        print(f"   tier C: utc_partition={tier_c_observation(line)}")
        print(f"   true business date: {truth}")
    print("   => tier B and tier C observations are IDENTICAL for A and B,")
    print("      but the true business dates differ (2025-03-04 vs 2025-04-03).")
    print("   => disambiguation requires retaining the raw matched date span")
    print("      ('2025-03-04' vs '03/04/2025') or a per-source convention tag")
    print("      (template id / dayfirst flag). Neither is in tier B or C.")


if __name__ == "__main__":
    main()
