"""Tests for the explicit site-consistency entry point."""

import datetime as _datetime
from datetime import datetime, timedelta

import pytest

from dateutil import tz
from dateutil.parser import (
    ConsistencyResult,
    InstantCandidate,
    ParserError,
    parse,
    parse_consistent,
    parser,
    parserinfo,
)

from ._common import TZEnvContext


UTC = tz.UTC
NYC = tz.gettz("America/New_York")
TORONTO = tz.gettz("America/Toronto")
CHICAGO = tz.gettz("America/Chicago")
LONDON = tz.gettz("Europe/London")


def _candidate(local, fold):
    return local.replace(tzinfo=NYC, fold=fold).astimezone(UTC)


def test_fall_back_keeps_both_candidates():
    # 01:30 occurs twice in New York on 2024-11-03.
    local = datetime(2024, 11, 3, 1, 30)
    result = parse_consistent("2024-11-03 01:30", sites={"NYC": NYC})

    assert result.status == "conflict"
    assert result.local == local
    assert len(result.candidates) == 2
    assert {candidate.fold for candidate in result.candidates} == {0, 1}
    assert result.instants == (
        _candidate(local, 0),
        _candidate(local, 1),
    )
    assert result.warehoused_instant is None
    assert result.review_required
    assert result.pending_sites == ("NYC",)


def test_spring_forward_gap_has_no_candidate():
    # 02:30 does not exist in New York on 2024-03-10.
    result = parse_consistent("2024-03-10 02:30", sites={"NYC": NYC})

    assert result.status == "no_candidate"
    assert result.candidates == ()
    assert result.site_candidates["NYC"] == ()
    assert result.instants == ()
    assert result.warehoused_instant is None


def test_abbreviation_selects_the_fall_back_occurrence():
    without = parse_consistent("2024-11-03 01:30 EST", sites={"NYC": NYC})
    with_offset = parse_consistent("2024-11-03 01:30 EST -0500",
                                   sites={"NYC": NYC})

    expected = _candidate(datetime(2024, 11, 3, 1, 30), 1)

    assert without.status == "resolved"
    assert without.instants == (expected,)
    assert without.evidence["abbreviation"] == "EST"
    assert without.evidence["offset"] is None
    assert "abbreviation" in without.critical_evidence
    assert "offset" in without.missing
    assert without.critical_sites == ("NYC",)
    assert without.load_bearing_evidence == without.critical_evidence
    assert without.utc_instants == without.instants

    assert with_offset.status == "resolved"
    assert with_offset.instants == (expected,)
    assert with_offset.evidence["abbreviation"] == "EST"
    assert with_offset.evidence["offset"] == -5 * 3600
    # Both pieces agree: removing either one leaves the same single answer.
    assert with_offset.critical_evidence == ()


def test_compact_offset_evidence_is_read_from_the_string():
    compact = parse_consistent("2024-11-03 01:30 EST-0500",
                               sites={"NYC": NYC})
    spaced = parse_consistent("2024-11-03 01:30 EST -0500",
                              sites={"NYC": NYC})

    assert compact.evidence == spaced.evidence
    assert compact.instants == spaced.instants == (
        _candidate(datetime(2024, 11, 3, 1, 30), 1),
    )


def test_offset_alone_selects_the_fall_back_occurrence():
    result = parse_consistent("2024-11-03 01:30 -0500", sites={"NYC": NYC})

    assert result.status == "resolved"
    assert result.evidence["offset"] == -5 * 3600
    assert result.evidence["abbreviation"] is None
    assert result.instants == (
        _candidate(datetime(2024, 11, 3, 1, 30), 1),
    )
    assert result.critical_evidence == ("offset",)


def test_contradictory_evidence_leaves_no_candidate():
    # EST is the post-transition abbreviation; -04:00 is the pre-transition
    # offset, so no instant in New York satisfies both.
    result = parse_consistent("2024-11-03 01:30 EST -0400",
                              sites={"NYC": NYC})

    assert result.status == "no_candidate"
    assert result.candidates == ()
    # Dropping either piece of evidence would resolve 01:30 to a single
    # occurrence, so each one changes the conclusion.
    assert result.critical_evidence == ("abbreviation", "offset")


def test_shared_instant_between_sites_is_warehoused_but_pending():
    result = parse_consistent("2024-11-03 01:30 EST",
                              sites={"NYC": NYC, "TORONTO": TORONTO})

    assert result.status == "site_pending"
    assert result.warehoused_instant == _candidate(
        datetime(2024, 11, 3, 1, 30), 1)
    assert result.instants == (result.warehoused_instant,)
    assert result.pending_sites == ("NYC", "TORONTO")
    assert not result.review_required

    # Both sites identify the same instant, so each is load-bearing only
    # through the candidate it contributes.
    assert result.critical_sites == ("NYC", "TORONTO")


def test_different_instants_are_handed_to_review():
    result = parse_consistent("2024-06-01 12:00",
                              sites={"NYC": NYC, "CHICAGO": CHICAGO})

    assert result.status == "conflict"
    assert len(result.instants) == 2
    assert result.warehoused_instant is None
    assert result.review_required


def test_missing_sites_do_not_use_the_host_timezone():
    result = parse_consistent("2024-06-01 12:00 EDT")

    assert result.status == "missing_sites"
    assert result.instants == ()
    assert result.warehoused_instant is None
    assert "sites" in result.missing
    assert "offset" in result.missing
    # The abbreviation was actually present.
    assert result.evidence["abbreviation"] == "EDT"
    assert "abbreviation" not in result.missing


def test_evidence_is_independent_of_host_timezone():
    def snapshot():
        return [
            parse_consistent(text, sites={"NYC": NYC}).as_dict()
            for text in (
                "2024-11-03 01:30 EST",
                "2024-03-10 02:30",
                "2024-06-01 12:00 EDT",
            )
        ]

    with TZEnvContext("EST5EDT,M3.2.0/2,M11.1.0/2"):
        first = snapshot()
    with TZEnvContext("NZST-12NZDT,M9.5.0,M4.1.0/3"):
        second = snapshot()

    assert first == second


def test_unknown_local_offset_is_missing_evidence():
    # "-0000" means the local offset is unknown; it must not be treated as
    # a known UTC offset.
    for text in ("2024-11-03 01:30 -0000", "2024-11-03 01:30 -00:00"):
        result = parse_consistent(text, sites={"NYC": NYC})
        assert result.status == "conflict"
        assert result.evidence["offset"] is None
        assert result.evidence["unknown_offset"] is True
        assert "offset" in result.missing


def test_explicit_utc_offset_is_known():
    result = parse_consistent("2024-11-03 01:30 +0000",
                              sites={"NYC": NYC})

    assert result.evidence["offset"] == 0
    assert result.evidence["unknown_offset"] is False
    assert result.status == "no_candidate"


def test_parse_routes_to_consistency_when_sites_are_given():
    routed = parse("2024-11-03 01:30 EST", sites={"NYC": NYC})
    explicit = parse_consistent("2024-11-03 01:30 EST",
                                sites={"NYC": NYC})

    assert isinstance(routed, ConsistencyResult)
    assert routed.instants == explicit.instants


def test_parse_consistent_switch_without_sites():
    result = parse("2024-06-01 12:00 EDT", consistent=True)

    assert isinstance(result, ConsistencyResult)
    assert result.status == "missing_sites"


def test_explicit_api_leaves_plain_parse_unchanged():
    text = "2024-11-03 01:30 EST -0500"
    parsed = parse(text)

    assert parsed == datetime(2024, 11, 3, 1, 30,
                              tzinfo=tz.tzoffset(None, -5 * 3600))


def test_sites_may_be_keys_objects_or_offsets():
    by_key = parse_consistent("2024-06-01 12:00 EDT",
                              sites=["America/New_York"])
    by_object = parse_consistent("2024-06-01 12:00 EDT", sites=[NYC])
    by_offset = parse_consistent("2024-06-01 12:00 EDT", sites=[-5 * 3600])

    assert by_key.status == by_object.status == "resolved"
    assert by_key.instants == by_object.instants
    assert by_key.candidates[0].site == "America/New_York"
    assert by_key.candidates[0].timezone.key == "America/New_York"
    assert by_object.candidates[0].site == "America/New_York"

    # A fixed-offset zone cannot carry an abbreviation, so EDT is rejected.
    assert by_offset.status == "no_candidate"
    assert by_offset.candidates == ()


def test_single_site_can_be_given_without_a_sequence():
    by_string = parse_consistent("2024-06-01 12:00 EDT",
                                 sites="America/New_York")
    by_zone = parse_consistent("2024-06-01 12:00 EDT", sites=NYC)

    assert by_string.status == by_zone.status == "resolved"
    assert by_string.instants == by_zone.instants


def test_site_labels_resolve_through_tzinfos():
    mapping = {"NYC": NYC, "CHI": CHICAGO}
    result = parse_consistent(
        "2024-06-01 12:00 EDT",
        sites=["NYC", "CHI"],
        tzinfos=mapping,
    )

    assert result.status == "resolved"
    assert result.evidence["sites"] == ("NYC", "CHI")
    assert result.candidates[0].site == "NYC"
    assert result.candidates[0].timezone is NYC


def test_site_labels_resolve_through_callable_tzinfos():
    def aliases(name, offset):
        return {"NYC": NYC, "CHI": CHICAGO}.get(name)

    result = parse_consistent("2024-06-01 12:00 EDT",
                              sites=["NYC", "CHI"], tzinfos=aliases)

    assert result.status == "resolved"
    assert result.candidates[0].timezone is NYC


def test_tzinfos_is_the_site_list_when_sites_are_omitted():
    result = parse_consistent("2024-06-01 12:00 EDT",
                              tzinfos={"NYC": NYC})

    assert result.status == "resolved"
    assert result.candidates[0].site == "NYC"
    assert result.evidence["sites"] == ("NYC",)


def test_site_labels_must_be_unique():
    with pytest.raises(ValueError):
        parse_consistent("2024-06-01 12:00",
                         sites=[("X", NYC), ("X", CHICAGO)])


def test_unknown_site_zone_is_rejected():
    with pytest.raises(ValueError):
        parse_consistent("2024-06-01 12:00", sites=["No/Such_Zone"])


def test_unknown_site_alias_is_rejected():
    with pytest.raises(ValueError):
        parse_consistent("2024-06-01 12:00", sites=["NOPE"],
                         tzinfos={"NYC": NYC})


def test_invalid_site_type_is_rejected():
    with pytest.raises(TypeError):
        parse_consistent("2024-06-01 12:00", sites=[object()])
    with pytest.raises(TypeError):
        parse_consistent("2024-06-01 12:00", sites=[True])


def test_bad_input_still_raises_parser_error():
    with pytest.raises(ParserError):
        parse_consistent("not a date", sites=[NYC])


def test_default_supplies_omitted_date_parts():
    default = datetime(2024, 11, 3)
    result = parse_consistent("01:30 EST", sites={"NYC": NYC},
                              default=default)

    assert result.local == datetime(2024, 11, 3, 1, 30)
    assert result.status == "resolved"


def test_parser_options_are_forwarded():
    result = parse_consistent("03/11/2024 01:30", sites={"NYC": NYC},
                              dayfirst=True)

    assert result.local == datetime(2024, 11, 3, 1, 30)


def test_custom_parser_instance_matches_wrapper():
    class _Info(parserinfo):
        pass

    info = _Info()
    direct = parser(info).parse_consistent(
        "2024-06-01 12:00 EDT", sites={"NYC": NYC})
    wrapped = parse_consistent(
        "2024-06-01 12:00 EDT", sites={"NYC": NYC}, parserinfo=info)

    assert isinstance(direct, ConsistencyResult)
    assert direct.instants == wrapped.instants


def test_candidate_exposes_the_site_rules():
    result = parse_consistent("2024-11-03 01:30 EST", sites={"NYC": NYC})
    candidate = result.candidates[0]

    assert isinstance(candidate, InstantCandidate)
    assert candidate.site == "NYC"
    assert candidate.instant.tzinfo is UTC
    assert candidate.local == datetime(2024, 11, 3, 1, 30)
    assert candidate.tzname == "EST"
    assert candidate.utcoffset == timedelta(hours=-5)
    assert candidate.offset_seconds == -5 * 3600
    assert candidate.timezone is NYC

    assert candidate == InstantCandidate(
        candidate.site, candidate.instant, candidate.local,
        candidate.tzname, candidate.utcoffset, candidate.fold,
        candidate.timezone)
    assert hash(candidate) == hash(result.candidates[0])
    assert candidate != "candidate"


def test_candidate_repr_contains_key_fields():
    result = parse_consistent("2024-11-03 01:30 EST", sites={"NYC": NYC})
    text = repr(result.candidates[0])

    assert "site='NYC'" in text
    assert "tzname='EST'" in text


def test_result_as_dict_keeps_evidence():
    result = parse_consistent("2024-11-03 01:30 EST-0500",
                              sites={"NYC": NYC})
    payload = result.as_dict()

    assert payload["status"] == "resolved"
    assert payload["warehoused_instant"] is not None
    assert payload["evidence"]["abbreviation"] == "EST"
    assert payload["evidence"]["offset"] == -5 * 3600
    assert payload["critical_evidence"] == []
    assert payload["critical_sites"] == ["NYC"]


def test_repr_of_result():
    result = parse_consistent("2024-11-03 01:30", sites={"NYC": NYC})
    assert "status='conflict'" in repr(result)


def test_mismatched_abbreviation_eliminates_the_site():
    result = parse_consistent("2024-06-01 12:00 EST", sites={"NYC": NYC})

    assert result.status == "no_candidate"
    assert result.candidates == ()
    assert result.evidence["abbreviation"] == "EST"


def test_site_without_a_candidate_does_not_hide_another_site():
    # 02:30 is a gap in New York, but it exists in London on that date.
    result = parse_consistent("2024-03-10 02:30 GMT",
                              sites={"NYC": NYC, "LONDON": LONDON})

    assert result.status == "resolved"
    assert result.compatible_sites == ("LONDON",)
    assert result.site_candidates["NYC"] == ()
    assert result.instants == (
        datetime(2024, 3, 10, 2, 30, tzinfo=UTC),
    )


def test_half_hour_dst_fold_keeps_both_offsets():
    # Lord Howe Island runs a 30-minute fall-back on 2024-04-07.
    zone = tz.gettz("Australia/Lord_Howe")
    result = parse_consistent("2024-04-07 01:45", sites={"LH": zone})

    assert result.status == "conflict"
    assert {candidate.offset_seconds for candidate in result.candidates} == {
        11 * 3600, 10 * 3600 + 30 * 60,
    }

    selected = parse_consistent("2024-04-07 01:45 +10:30", sites={"LH": zone})
    assert selected.status == "resolved"
    assert selected.candidates[0].fold == 1


def test_skipped_whole_day_has_no_candidates():
    # Samoa skipped 2011-12-30 entirely when it moved across the date line.
    zone = tz.gettz("Pacific/Apia")
    result = parse_consistent("2011-12-30 12:00 +14:00", sites={"APIA": zone})

    assert result.status == "no_candidate"
    assert result.instants == ()


def test_tzinfo_without_offsets_is_skipped():
    class _Naive(_datetime.tzinfo):
        pass

    result = parse_consistent("2024-06-01 12:00", sites={"NAIVE": _Naive()})

    assert result.status == "no_candidate"
    assert result.site_candidates["NAIVE"] == ()


def test_zulu_suffix_is_reported_as_utc_abbreviation():
    result = parse_consistent("2024-06-01T12:00:00Z", sites={"UTC": UTC})

    assert result.evidence["abbreviation"] == "UTC"
    assert result.status == "resolved"
    assert result.instants == (datetime(2024, 6, 1, 12, tzinfo=UTC),)


def test_alias_resolvers_may_decline_a_label():
    def no_alias(name, offset):
        return None

    with pytest.raises(ValueError):
        parse_consistent("2024-06-01 12:00", sites=["NYC"],
                         tzinfos=no_alias)
    with pytest.raises(ValueError):
        parse_consistent("2024-06-01 12:00", sites=["NYC"],
                         tzinfos={"NYC": None})


def test_out_of_range_offset_in_input_is_ignored():
    # The lexer accepts it as a number, but it is not a valid zone offset.
    result = parse_consistent("2024-06-01 12:00 +2500", sites={"NYC": NYC})

    assert result.evidence["offset"] is None


def test_fuzzy_input_is_forwarded():
    result = parse_consistent("Meeting at 2024-11-03 01:30 EST",
                              sites={"NYC": NYC}, fuzzy=True)

    assert result.local == datetime(2024, 11, 3, 1, 30)
    assert result.status == "resolved"
