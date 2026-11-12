"""
Regression harness for ``rrulestr`` upgrades.

The suite evaluates calendar configs (bounded ``DTSTART``/``RRULE``/
``RDATE``/``EXDATE`` plus line folding) twice:

* an independent oracle enumerates the expected partition (occurrence
  set) over a small date domain, tracking the source line of every
  occurrence and exclusion;
* ``rrulestr`` parses the same text and its in-domain occurrences are
  compared against the oracle partition.

Comparisons never stop at the first few outputs: both sides are fully
enumerated over the domain, so equal sets prove in-domain equivalence
and unequal sets report the earliest divergence.  Verdicts distinguish
a parser rejection, an item that was accepted but ignored, and
different spellings that are set-equivalent.
"""

import enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from dateutil.rrule import rrulestr

DOMAIN_START = datetime(2026, 1, 1)
DOMAIN_END = datetime(2026, 2, 1)

_FREQ_STEPS = {
    "SECONDLY": timedelta(seconds=1),
    "MINUTELY": timedelta(minutes=1),
    "HOURLY": timedelta(hours=1),
    "DAILY": timedelta(days=1),
    "WEEKLY": timedelta(weeks=1),
}

_MAX_EXPANSION = 10000


class OracleError(ValueError):
    pass


def _unfold(text):
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line[0] == " " and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)
    return lines


def _parse_dt(value):
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S")
    except ValueError:
        raise OracleError("unsupported datetime: %r" % value)


@dataclass
class OraclePartition:
    occurrences: dict = field(default_factory=dict)  # dt -> [source tags]
    exclusions: dict = field(default_factory=dict)   # dt -> [source tags]
    unaccounted: list = field(default_factory=list)  # (tag, content)
    partition: frozenset = frozenset()


def _expand_rrule(value, dtstart, tag, unaccounted):
    kwargs = {}
    for pair in value.split(";"):
        if "=" not in pair:
            unaccounted.append((tag, "trailing garbage: %s" % pair))
            continue
        key, _, val = pair.partition("=")
        key = key.upper()
        if key in ("FREQ", "INTERVAL", "COUNT", "UNTIL"):
            kwargs[key] = val.upper()
        else:
            unaccounted.append((tag, "unknown RRULE part: %s" % pair))

    freq = kwargs.get("FREQ")
    if freq not in _FREQ_STEPS:
        raise OracleError("unsupported FREQ: %r" % freq)
    interval = int(kwargs.get("INTERVAL", "1"))
    count = int(kwargs["COUNT"]) if "COUNT" in kwargs else None
    until = _parse_dt(kwargs["UNTIL"]) if "UNTIL" in kwargs else None
    if count is None and until is None:
        raise OracleError("unbounded RRULE: %s" % value)

    step = _FREQ_STEPS[freq] * interval
    occurrences = []
    current = dtstart
    for _ in range(_MAX_EXPANSION):
        if until is not None and current > until:
            break
        occurrences.append(current)
        if count is not None and len(occurrences) >= count:
            break
        current += step
    else:
        raise OracleError("RRULE expansion exceeded safety bound")
    return occurrences


def oracle_partition(text, domain_start=DOMAIN_START, domain_end=DOMAIN_END):
    """Independently enumerate the expected in-domain partition.

    Every occurrence and exclusion is tagged with the source line that
    produced it; input outside the bounded subset is recorded as
    unaccounted instead of being silently dropped.
    """
    result = OraclePartition()
    dtstart = None
    pending = []

    for lineno, line in enumerate(_unfold(text), start=1):
        if ":" not in line:
            result.unaccounted.append(("L%d" % lineno, line))
            continue
        name, _, value = line.partition(":")
        parts = name.split(";")
        prop = parts[0].upper()
        parms = parts[1:]
        tag = "%s#L%d" % (prop, lineno)

        if prop == "DTSTART":
            dtstart = _parse_dt(value)
        elif prop == "RRULE":
            for parm in parms:
                result.unaccounted.append((tag, "RRULE parm: %s" % parm))
            pending.append(("RRULE", tag, value))
        elif prop == "RDATE":
            for datestr in value.split(","):
                pending.append(("RDATE", tag, datestr))
        elif prop == "EXDATE":
            for datestr in value.split(","):
                dt = _parse_dt(datestr)
                result.exclusions.setdefault(dt, []).append(tag)
        else:
            result.unaccounted.append((tag, line))

    for kind, tag, value in pending:
        if kind == "RRULE":
            if dtstart is None:
                raise OracleError("RRULE without DTSTART")
            dts = _expand_rrule(value, dtstart, tag, result.unaccounted)
        else:
            dts = [_parse_dt(value)]
        for dt in dts:
            result.occurrences.setdefault(dt, []).append(tag)

    excluded = set(result.exclusions)
    result.partition = frozenset(
        dt
        for dt in result.occurrences
        if dt not in excluded and domain_start <= dt <= domain_end
    )
    return result


@dataclass
class ParserOutcome:
    rejected: bool
    error: Exception = None
    occurrences: frozenset = None


def evaluate_config(text, domain_start=DOMAIN_START, domain_end=DOMAIN_END):
    try:
        rset = rrulestr(text, forceset=True, unfold=True)
    except Exception as exc:
        return ParserOutcome(rejected=True, error=exc)
    occurrences = frozenset(rset.between(domain_start, domain_end, inc=True))
    return ParserOutcome(rejected=False, occurrences=occurrences)


@dataclass
class PartitionDiff:
    equivalent: bool
    divergence: datetime = None
    only_in_a: frozenset = frozenset()
    only_in_b: frozenset = frozenset()


def diff_partitions(set_a, set_b):
    """Compare two fully enumerated in-domain partitions.

    Equal sets prove equivalence over the whole domain; otherwise the
    earliest disagreeing datetime is reported, even when the two
    configurations share a long common prefix of outputs.
    """
    if set_a == set_b:
        return PartitionDiff(equivalent=True)
    only_in_a = set_a - set_b
    only_in_b = set_b - set_a
    return PartitionDiff(
        equivalent=False,
        divergence=min(only_in_a | only_in_b),
        only_in_a=only_in_a,
        only_in_b=only_in_b,
    )


class Verdict(enum.Enum):
    MATCHES_ORACLE = "matches_oracle"
    PARSER_REJECTED = "parser_rejected"
    ACCEPTED_BUT_IGNORED = "accepted_but_ignored"
    PARTITION_CHANGED = "partition_changed"


@dataclass
class Classification:
    verdict: Verdict
    oracle: OraclePartition
    outcome: ParserOutcome
    diff: PartitionDiff = None
    ignored: dict = field(default_factory=dict)


def classify_config(text, domain_start=DOMAIN_START, domain_end=DOMAIN_END):
    oracle = oracle_partition(text, domain_start, domain_end)
    outcome = evaluate_config(text, domain_start, domain_end)
    if outcome.rejected:
        return Classification(Verdict.PARSER_REJECTED, oracle, outcome)

    diff = diff_partitions(outcome.occurrences, oracle.partition)
    if diff.equivalent:
        verdict = (
            Verdict.ACCEPTED_BUT_IGNORED
            if oracle.unaccounted
            else Verdict.MATCHES_ORACLE
        )
        return Classification(verdict, oracle, outcome, diff)

    missing = oracle.partition - outcome.occurrences
    if missing and not (outcome.occurrences - oracle.partition):
        verdict = Verdict.ACCEPTED_BUT_IGNORED
    else:
        verdict = Verdict.PARTITION_CHANGED
    ignored = {
        dt: oracle.occurrences[dt] for dt in sorted(missing)
    }
    return Classification(verdict, oracle, outcome, diff, ignored)


class ConfigVerdict(enum.Enum):
    BOTH_REJECTED = "both_rejected"
    REJECTION_MISMATCH = "rejection_mismatch"
    SET_EQUIVALENT = "set_equivalent"
    PARTITION_CHANGED = "partition_changed"


@dataclass
class ConfigComparison:
    verdict: ConfigVerdict
    outcome_a: ParserOutcome
    outcome_b: ParserOutcome
    diff: PartitionDiff = None
    domain: tuple = (DOMAIN_START, DOMAIN_END)


def compare_configs(text_a, text_b,
                    domain_start=DOMAIN_START, domain_end=DOMAIN_END):
    outcome_a = evaluate_config(text_a, domain_start, domain_end)
    outcome_b = evaluate_config(text_b, domain_start, domain_end)
    domain = (domain_start, domain_end)

    if outcome_a.rejected or outcome_b.rejected:
        verdict = (
            ConfigVerdict.BOTH_REJECTED
            if outcome_a.rejected and outcome_b.rejected
            else ConfigVerdict.REJECTION_MISMATCH
        )
        return ConfigComparison(verdict, outcome_a, outcome_b, domain=domain)

    diff = diff_partitions(outcome_a.occurrences, outcome_b.occurrences)
    verdict = (
        ConfigVerdict.SET_EQUIVALENT
        if diff.equivalent
        else ConfigVerdict.PARTITION_CHANGED
    )
    return ConfigComparison(verdict, outcome_a, outcome_b, diff, domain)


BASELINE = (
    "DTSTART:20260105T090000\n"
    "RRULE:FREQ=DAILY;INTERVAL=2;COUNT=5\n"
    "RDATE:20260120T090000,20260122T090000\n"
    "EXDATE:20260109T090000\n"
)

BASELINE_PARTITION = frozenset(
    datetime(2026, 1, day, 9) for day in (5, 7, 11, 13, 20, 22)
)


@pytest.mark.rrulestr
class TestOraclePartition:
    def test_baseline_partition(self):
        oracle = oracle_partition(BASELINE)
        assert oracle.partition == BASELINE_PARTITION
        assert not oracle.unaccounted

    def test_provenance_tracks_source_lines(self):
        oracle = oracle_partition(BASELINE)
        assert oracle.occurrences[datetime(2026, 1, 5, 9)] == ["RRULE#L2"]
        assert oracle.occurrences[datetime(2026, 1, 20, 9)] == ["RDATE#L3"]
        assert oracle.exclusions[datetime(2026, 1, 9, 9)] == ["EXDATE#L4"]
        assert datetime(2026, 1, 9, 9) not in oracle.partition

    def test_duplicate_sources_all_recorded(self):
        text = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=3\n"
            "RDATE:20260106T090000\n"
        )
        oracle = oracle_partition(text)
        assert oracle.occurrences[datetime(2026, 1, 6, 9)] == [
            "RRULE#L2",
            "RDATE#L3",
        ]

    def test_unbounded_rule_rejected(self):
        with pytest.raises(OracleError):
            oracle_partition("DTSTART:20260105T090000\nRRULE:FREQ=DAILY")


@pytest.mark.rrulestr
class TestClassifyConfig:
    def test_baseline_matches_oracle(self):
        result = classify_config(BASELINE)
        assert result.verdict is Verdict.MATCHES_ORACLE
        assert result.outcome.occurrences == BASELINE_PARTITION

    def test_folded_input_matches_oracle(self):
        folded = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAI\n"
            " LY;INTER\n"
            " VAL=2;COUNT=5\n"
            "RDATE:20260120T090000,2026\n"
            " 0122T090000\n"
            "EXDATE:20260109T090000\n"
        )
        result = classify_config(folded)
        assert result.verdict is Verdict.MATCHES_ORACLE
        assert result.outcome.occurrences == BASELINE_PARTITION

    def test_unknown_property_line_rejected(self):
        # The unknown property sits next to the rule that would produce
        # the later occurrences, so a rejection loses that partition.
        text = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=2\n"
            "X-UNKNOWN-PROP:late\n"
            "RRULE:FREQ=WEEKLY;COUNT=3\n"
        )
        result = classify_config(text)
        assert result.verdict is Verdict.PARSER_REJECTED
        assert datetime(2026, 1, 12, 9) in result.oracle.partition
        assert datetime(2026, 1, 19, 9) in result.oracle.partition
        assert any("X-UNKNOWN-PROP" in item for _, item in
                   result.oracle.unaccounted)

    def test_unknown_rrule_parm_rejected(self):
        text = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=2\n"
            "RRULE;SOMEPARM=1:FREQ=WEEKLY;COUNT=2\n"
        )
        result = classify_config(text)
        assert result.verdict is Verdict.PARSER_REJECTED
        assert datetime(2026, 1, 12, 9) in result.oracle.partition

    def test_trailing_garbage_rejected(self):
        # Garbage trails the rule that would produce the later
        # occurrences; ignoring it would change the partition.
        text = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=2\n"
            "RRULE:FREQ=WEEKLY;COUNT=2;GARBAGE\n"
        )
        result = classify_config(text)
        assert result.verdict is Verdict.PARSER_REJECTED
        assert datetime(2026, 1, 12, 9) in result.oracle.partition
        assert any("GARBAGE" in item for _, item in
                   result.oracle.unaccounted)

    def test_x_parm_accepted_but_ignored(self):
        text = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=2\n"
            "RRULE;X-VOORAY=1:FREQ=WEEKLY;COUNT=2\n"
        )
        result = classify_config(text)
        assert result.verdict is Verdict.ACCEPTED_BUT_IGNORED
        assert datetime(2026, 1, 12, 9) in result.outcome.occurrences
        assert result.outcome.occurrences == result.oracle.partition


@pytest.mark.rrulestr
class TestCompareConfigs:
    def test_folding_is_set_equivalent(self):
        unfolded = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;INTERVAL=2;COUNT=5\n"
        )
        folded = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAI\n"
            " LY;INTER\n"
            " VAL=2;COUNT=5\n"
        )
        comparison = compare_configs(folded, unfolded)
        assert comparison.verdict is ConfigVerdict.SET_EQUIVALENT

    def test_duplicate_exdate_is_set_invariant(self):
        single = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=4\n"
            "EXDATE:20260107T090000\n"
        )
        duplicate = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=4\n"
            "EXDATE:20260107T090000\n"
            "EXDATE:20260107T090000\n"
        )
        comparison = compare_configs(duplicate, single)
        assert comparison.verdict is ConfigVerdict.SET_EQUIVALENT

        oracle = oracle_partition(duplicate)
        assert oracle.exclusions[datetime(2026, 1, 7, 9)] == [
            "EXDATE#L3",
            "EXDATE#L4",
        ]
        assert classify_config(duplicate).verdict is Verdict.MATCHES_ORACLE

    def test_earliest_divergence_after_common_prefix(self):
        # Both configs produce 2026-01-05, 07, 09, 11 first; they only
        # diverge afterwards.
        text_a = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;INTERVAL=2;COUNT=6\n"
        )
        text_b = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;INTERVAL=2;COUNT=3\n"
            "RDATE:20260111T090000,20260112T090000,20260113T090000\n"
        )
        comparison = compare_configs(text_a, text_b)
        assert comparison.verdict is ConfigVerdict.PARTITION_CHANGED

        diff = comparison.diff
        assert diff.divergence == datetime(2026, 1, 12, 9)
        assert diff.only_in_b == frozenset({datetime(2026, 1, 12, 9)})
        assert diff.only_in_a == frozenset({datetime(2026, 1, 15, 9)})

        common_prefix = [
            dt for dt in sorted(comparison.outcome_a.occurrences)
            if dt < diff.divergence
        ]
        assert common_prefix == [
            datetime(2026, 1, day, 9) for day in (5, 7, 9, 11)
        ]
        assert common_prefix == [
            dt for dt in sorted(comparison.outcome_b.occurrences)
            if dt < diff.divergence
        ]

    def test_equivalent_spellings_proven_over_domain(self):
        by_count = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=4\n"
        )
        by_until = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;UNTIL=20260108T090000\n"
        )
        by_rdates = (
            "DTSTART:20260105T090000\n"
            "RDATE:20260105T090000,20260106T090000\n"
            "RDATE:20260107T090000,20260108T090000\n"
        )
        weekly = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=WEEKLY;COUNT=3\n"
        )
        daily_interval_7 = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;INTERVAL=7;COUNT=3\n"
        )

        for text_b in (by_until, by_rdates):
            comparison = compare_configs(by_count, text_b)
            assert comparison.verdict is ConfigVerdict.SET_EQUIVALENT
            assert comparison.domain == (DOMAIN_START, DOMAIN_END)

        comparison = compare_configs(weekly, daily_interval_7)
        assert comparison.verdict is ConfigVerdict.SET_EQUIVALENT

    def test_rejection_mismatch(self):
        accepted = (
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=DAILY;COUNT=2\n"
        )
        rejected = accepted + "X-UNKNOWN-PROP:late\n"
        comparison = compare_configs(accepted, rejected)
        assert comparison.verdict is ConfigVerdict.REJECTION_MISMATCH
        assert not comparison.outcome_a.rejected
        assert comparison.outcome_b.rejected

        both = compare_configs(rejected, rejected)
        assert both.verdict is ConfigVerdict.BOTH_REJECTED
