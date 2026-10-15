"""Reading the SSML in a reply."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def _parse(ssml: str) -> ET.Element:
    try:
        return ET.fromstring(ssml)
    except ET.ParseError as exc:
        raise ValueError(f"not well-formed SSML ({exc}): {ssml!r}") from None


def well_formed(ssml: str) -> bool:
    """True when ``ssml`` parses as XML with a single <speak> root."""
    try:
        return _parse(ssml).tag == "speak"
    except ValueError:
        return False


def ssml_tags(ssml: str) -> set[str]:
    """Every tag name used in ``ssml``, the root included. ValueError if not well-formed."""
    return {element.tag for element in _parse(ssml).iter()}


def spoken_text(ssml: str) -> str:
    """The words the speech engine reads, tags removed, entities decoded, spaces collapsed.

    ValueError if not well-formed.
    """
    return " ".join("".join(_parse(ssml).itertext()).split())


def say_as(ssml: str, interpret_as: str) -> list[str]:
    """The text inside each <say-as interpret-as="..."> element. ValueError if not well-formed."""
    return [
        "".join(element.itertext())
        for element in _parse(ssml).iter("say-as")
        if element.get("interpret-as") == interpret_as
    ]
