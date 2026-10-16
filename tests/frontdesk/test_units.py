"""Unit checks of the parsing and speech helpers, no server."""

from __future__ import annotations

from datetime import date, time

import pytest

from frontdesk.parsing import find_phone_digits, parse_day, parse_dob, parse_time
from frontdesk.speech import ordinal, spoken_slot, spoken_time
from frontdesk.world import clinic_time, is_staffed

WEDNESDAY = date(2026, 10, 14)


@pytest.mark.parametrize(
    "span",
    ["1980-03-03", "3/3/80", "03/03/1980", "March 3rd, 1980", "March 3 1980", "3rd March 1980"],
)
def test_parse_dob_accepts_the_forms_callers_use(span):
    assert parse_dob(span, 2026) == date(1980, 3, 3)


def test_parse_dob_needs_a_year():
    assert parse_dob("March 3rd", 2026) is None


@pytest.mark.parametrize(
    ("said", "day"),
    [
        ("next Tuesday", date(2026, 10, 20)),
        ("Tuesday", date(2026, 10, 20)),
        ("next Wednesday", date(2026, 10, 21)),
        ("tomorrow", date(2026, 10, 15)),
        ("October 20th", date(2026, 10, 20)),
        ("10/20", date(2026, 10, 20)),
    ],
)
def test_parse_day_is_relative_to_the_call(said, day):
    assert parse_day(said, WEDNESDAY) == day


@pytest.mark.parametrize(
    ("said", "clock"),
    [("9:30", time(9, 30)), ("at 2", time(14, 0)), ("11 am", time(11, 0)), ("noon", time(12))],
)
def test_parse_time(said, clock):
    assert parse_time(said) == clock


@pytest.mark.parametrize(
    ("said", "digits"),
    [
        ("two one five, five five five, zero one four two", "2155550142"),
        ("two one five 555 oh one four two", "2155550142"),
        ("call me at 215-555-0142", "2155550142"),
    ],
)
def test_phone_digits_said_as_words_or_digits(said, digits):
    assert find_phone_digits(said) == digits


@pytest.mark.parametrize(
    ("day", "text"), [(1, "1st"), (2, "2nd"), (3, "3rd"), (11, "11th"), (22, "22nd"), (13, "13th")]
)
def test_ordinal(day, text):
    assert ordinal(day) == text


def test_spoken_slot_and_time():
    assert spoken_slot(clinic_time(2026, 10, 20, 9, 30)) == "Tuesday, October 20th at 9:30 AM"
    assert spoken_time(clinic_time(2026, 10, 20, 12)) == "12 PM"


@pytest.mark.parametrize(
    ("moment", "staffed"),
    [
        (clinic_time(2026, 10, 14, 8), True),
        (clinic_time(2026, 10, 14, 16, 59), True),
        (clinic_time(2026, 10, 14, 17), False),
        (clinic_time(2026, 10, 17, 10), False),
    ],
)
def test_staffed_hours(moment, staffed):
    assert is_staffed(moment) is staffed


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        ("3/3/26", date(2026, 3, 3)),
        ("3/3/27", date(1927, 3, 3)),
        ("1/1/00", date(2000, 1, 1)),
        ("12/31/99", date(1999, 12, 31)),
    ],
)
def test_two_digit_years_pivot_on_the_call_year(span, expected):
    assert parse_dob(span, 2026) == expected


@pytest.mark.parametrize(
    "span", ["February 30th 1980", "13/13/1980", "2026-02-30", "Smarch 3rd 1980", "the 3rd"]
)
def test_impossible_dates_are_not_dates(span):
    assert parse_dob(span, 2026) is None


def test_a_date_already_past_this_year_means_next_year():
    assert parse_day("October 1st", WEDNESDAY) == date(2027, 10, 1)
    assert parse_day("October 1st 2026", WEDNESDAY) == date(2026, 10, 1)
    assert parse_day("today", WEDNESDAY) == WEDNESDAY


@pytest.mark.parametrize(
    ("said", "clock"),
    [
        ("2 pm", time(14, 0)),
        ("12 am", time(0, 0)),
        ("12 pm", time(12, 0)),
        ("at 9", time(9, 0)),
        ("7:30 p.m.", time(19, 30)),
    ],
)
def test_parse_time_reads_the_meridiem(said, clock):
    assert parse_time(said) == clock


@pytest.mark.parametrize("said", ["25:00", "9:75", "at 25"])
def test_impossible_clock_values_are_not_times(said):
    assert parse_time(said) is None
