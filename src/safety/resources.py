"""
MWIZERWA AI — Emergency Resource Registry
Defines verified, toll-free emergency contacts in Rwanda and lookup helpers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class VerificationStatus(str, Enum):
    """Indicates the operational verification level of an emergency contact."""
    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class EmergencyContact:
    """
    Immutable representation of an emergency resource contact.
    Every emergency contact must have verified authority provenance.
    """
    phone_number: str
    service_name: str
    intended_use: str
    geographic_coverage: str
    verification_source: str
    verification_date: str
    verification_status: VerificationStatus
    description_rw: str
    description_en: str
    description_fr: str


# Authoritative, verified Rwanda emergency resources.
# 112: Rwanda National Police Emergency Hotline (General emergency, physical safety, immediate danger)
# 912: SAMU Ambulance / Medical Emergency (Acute medical, pre-hospital trauma, 24/7 ambulance dispatch)
VERIFIED_EMERGENCY_RESOURCES: Tuple[EmergencyContact, ...] = (
    EmergencyContact(
        phone_number="112",
        service_name="Rwanda National Police (RNP)",
        intended_use="General police emergencies, acute physical danger, violent threats, and crime response",
        geographic_coverage="Rwanda (toll-free nationwide)",
        verification_source="https://www.police.gov.rw/home/?page_id=35",
        verification_date="2026-11-18",
        verification_status=VerificationStatus.VERIFIED,
        description_rw="Nimero y'ubutabazi bwihuse ya Polisi y'u Rwanda (ku buntu)",
        description_en="Toll-free emergency line of the Rwanda National Police",
        description_fr="Ligne d'urgence sans frais de la Police Nationale du Rwanda",
    ),
    EmergencyContact(
        phone_number="912",
        service_name="SAMU Emergency Medical Services",
        intended_use="Acute pre-hospital medical emergencies, severe trauma, and 24/7 ambulance dispatch",
        geographic_coverage="Rwanda (toll-free nationwide)",
        verification_source="https://www.rbc.gov.rw/who-we-are/our-divisions-and-units/samu-services",
        verification_date="2026-11-18",
        verification_status=VerificationStatus.VERIFIED,
        description_rw="Ubutabazi bw'ibanze mu buvuzi n'imbangukiragutabara ya SAMU (ku buntu)",
        description_en="Toll-free 24/7 medical emergency and ambulance dispatch service (SAMU)",
        description_fr="Service d'aide médicale d'urgence et ambulances de SAMU (sans frais, 24/7)",
    ),
)


def get_all_resources() -> Tuple[EmergencyContact, ...]:
    """Return all emergency contacts in the registry."""
    return VERIFIED_EMERGENCY_RESOURCES


def get_verified_resources() -> Tuple[EmergencyContact, ...]:
    """Return only contacts with VERIFIED status."""
    return tuple(
        contact for contact in VERIFIED_EMERGENCY_RESOURCES
        if contact.verification_status == VerificationStatus.VERIFIED
    )


def get_emergency_contact_by_number(number: str) -> Optional[EmergencyContact]:
    """
    Look up an emergency contact by telephone number string.
    Normalizes whitespace and common formatting.
    """
    cleaned = number.strip().replace(" ", "").replace("-", "")
    for contact in VERIFIED_EMERGENCY_RESOURCES:
        if contact.phone_number == cleaned:
            return contact
    return None
