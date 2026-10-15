"""The seeded world: one fictional clinic, its providers, patients and schedule.

Every call gets its own fresh copy (``World.fresh()``), so calls never share
state and tests can run in parallel. All names, dates and numbers are invented.

Schedule: weekdays from Monday 2026-10-12 to Friday 2026-11-06, 30-minute
slots starting 08:00 to 16:30 clinic time (America/New_York). A slot is open
unless a patient appointment or a booking by someone outside the call holds it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

CLINIC_TZ = ZoneInfo("America/New_York")
SLOT = timedelta(minutes=30)
SCHEDULE_FIRST_DAY = date(2026, 10, 12)
SCHEDULE_LAST_DAY = date(2026, 11, 6)
DAY_FIRST_SLOT = time(8, 0)
DAY_LAST_SLOT = time(16, 30)
STAFFED_OPEN = time(8, 0)
STAFFED_CLOSE = time(17, 0)


def clinic_time(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """An aware datetime in clinic time."""
    return datetime(year, month, day, hour, minute, tzinfo=CLINIC_TZ)


@dataclass(frozen=True)
class Provider:
    id: str
    name: str  # as spoken: "Dr. Okafor"


@dataclass(frozen=True)
class Patient:
    id: str
    first_name: str
    last_name: str
    dob: date
    phone: str  # ten digits

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass
class Appointment:
    id: str
    patient_id: str
    provider_id: str
    start: datetime

    @property
    def end(self) -> datetime:
        return self.start + SLOT


PROVIDERS: tuple[Provider, ...] = (
    Provider("okafor", "Dr. Okafor"),
    Provider("lindqvist", "Dr. Lindqvist"),
    Provider("patel", "Dr. Patel"),
)

PATIENTS: tuple[Patient, ...] = (
    Patient("P-1001", "Maria", "Alvarez", date(1980, 3, 3), "2155550142"),
    Patient("P-1002", "James", "Whitaker", date(1975, 11, 9), "2155550177"),
    Patient("P-1003", "Priya", "Raman", date(1992, 7, 21), "2155550199"),
    Patient("P-1004", "Thomas", "Brennan", date(1968, 1, 15), "2155550123"),
)

# Appointments that exist before any call. The first one is in the past.
SEED_APPOINTMENTS: tuple[Appointment, ...] = (
    Appointment("A-0001", "P-1001", "lindqvist", clinic_time(2026, 9, 28, 10)),
    Appointment("A-0002", "P-1001", "lindqvist", clinic_time(2026, 10, 22, 14)),
    Appointment("A-0003", "P-1002", "patel", clinic_time(2026, 10, 20, 9, 30)),
    Appointment("A-0004", "P-1003", "okafor", clinic_time(2026, 10, 27, 9)),
)


def _block(provider_id: str, day: date, first: time, count: int) -> set[tuple[str, datetime]]:
    start = datetime.combine(day, first, tzinfo=CLINIC_TZ)
    return {(provider_id, start + SLOT * i) for i in range(count)}


# Slots held by patients outside the seeded list.
SEED_BUSY: frozenset[tuple[str, datetime]] = frozenset(
    _block("okafor", date(2026, 10, 20), time(8, 0), 6)  # Tue Oct 20, 08:00 to 10:30
    | _block("patel", date(2026, 10, 19), time(8, 0), 8)  # Mon Oct 19, 08:00 to 11:30
)


def schedule_days() -> Iterator[date]:
    day = SCHEDULE_FIRST_DAY
    while day <= SCHEDULE_LAST_DAY:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def day_slots(day: date) -> list[datetime]:
    start = datetime.combine(day, DAY_FIRST_SLOT, tzinfo=CLINIC_TZ)
    last = datetime.combine(day, DAY_LAST_SLOT, tzinfo=CLINIC_TZ)
    slots = []
    while start <= last:
        slots.append(start)
        start += SLOT
    return slots


def is_staffed(moment: datetime) -> bool:
    """Monday to Friday, 8:00 to 17:00 clinic time."""
    local = moment.astimezone(CLINIC_TZ)
    return local.weekday() < 5 and STAFFED_OPEN <= local.time() < STAFFED_CLOSE


@dataclass
class World:
    patients: dict[str, Patient]
    appointments: list[Appointment]
    busy: set[tuple[str, datetime]]
    next_appointment_number: int = 1
    _slots: dict[str, list[datetime]] = field(default_factory=dict, repr=False)

    @classmethod
    def fresh(cls) -> World:
        """A private copy of the seeded world for one call."""
        return cls(
            patients={p.id: p for p in PATIENTS},
            appointments=[copy.copy(a) for a in SEED_APPOINTMENTS],
            busy=set(SEED_BUSY),
        )

    @staticmethod
    def provider(provider_id: str) -> Provider:
        for provider in PROVIDERS:
            if provider.id == provider_id:
                return provider
        raise KeyError(provider_id)

    @staticmethod
    def find_provider(name: str) -> Provider | None:
        """Match "okafor", "Okafor" or "Dr. Okafor"."""
        key = name.strip().lower().removeprefix("dr.").removeprefix("dr ").strip()
        for provider in PROVIDERS:
            if key in (provider.id, provider.name.lower()):
                return provider
        return None

    def find_patients(self, full_name: str, dob: date) -> list[Patient]:
        wanted = " ".join(full_name.lower().split())
        return [p for p in self.patients.values() if p.full_name.lower() == wanted and p.dob == dob]

    def all_slots(self) -> list[datetime]:
        if "all" not in self._slots:
            self._slots["all"] = [slot for day in schedule_days() for slot in day_slots(day)]
        return self._slots["all"]

    def is_slot(self, start: datetime) -> bool:
        local = start.astimezone(CLINIC_TZ)
        return local.date() in set(schedule_days()) and local in day_slots(local.date())

    def _held(self, provider_id: str, start: datetime) -> list[Appointment]:
        return [a for a in self.appointments if a.provider_id == provider_id and a.start == start]

    def is_open(self, provider_id: str, start: datetime) -> bool:
        """Open: a real slot, not held by an appointment, not booked by someone else."""
        return (
            self.is_slot(start)
            and (provider_id, start) not in self.busy
            and not self._held(provider_id, start)
        )

    def taken_by_someone_else(self, provider_id: str, start: datetime, patient_id: str) -> bool:
        if (provider_id, start) in self.busy:
            return True
        return any(a.patient_id != patient_id for a in self._held(provider_id, start))

    def open_slots(self, provider_id: str, after: datetime) -> list[datetime]:
        return [s for s in self.all_slots() if s > after and self.is_open(provider_id, s)]

    def book(self, patient_id: str, provider_id: str, start: datetime) -> Appointment:
        appointment = Appointment(
            id=f"A-{1000 + self.next_appointment_number}",
            patient_id=patient_id,
            provider_id=provider_id,
            start=start,
        )
        self.next_appointment_number += 1
        self.appointments.append(appointment)
        return appointment

    def take(self, provider_id: str, start: datetime) -> None:
        """Test hook: someone outside this call books the slot."""
        self.busy.add((provider_id, start))

    def upcoming(self, patient_id: str | None, now: datetime) -> list[Appointment]:
        """Appointments after ``now``, soonest first; all patients when patient_id is None."""
        found = [
            a
            for a in self.appointments
            if a.start > now and (patient_id is None or a.patient_id == patient_id)
        ]
        return sorted(found, key=lambda a: a.start)
