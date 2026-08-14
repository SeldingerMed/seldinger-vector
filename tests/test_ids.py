"""Identifier minting and prefix enforcement."""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel, ValidationError

from or_audit.domain.ids import (
    EpisodeId,
    SurgeonId,
    mint,
    new_episode_id,
    new_surgeon_id,
)


class _Holder(BaseModel):
    episode_id: EpisodeId
    surgeon_id: SurgeonId


def test_minted_id_has_prefix_and_fixed_width():
    value = mint("epi")
    assert re.fullmatch(r"epi_[0-9A-HJKMNP-TV-Z]{26}", value)


def test_ids_are_unique_across_many_mints():
    values = {new_episode_id() for _ in range(2000)}
    assert len(values) == 2000


def test_ids_minted_later_sort_later():
    """Timestamp prefix must make identifiers sort in creation order."""
    first = mint("epi")
    # Same millisecond is possible, so compare the timestamp segment only
    # after forcing a tick.
    import time

    time.sleep(0.002)
    second = mint("epi")
    assert first < second


def test_ambiguous_characters_are_excluded():
    """Crockford base32 drops I, L, O and U to survive transcription."""
    body = mint("epi").split("_", 1)[1]
    assert not (set(body) & set("ILOU"))


@pytest.mark.parametrize("bad", ["", "EPI", "ep1", "ep_i", "épi", "xyz", "mrn"])
def test_unknown_or_malformed_prefix_rejected(bad):
    """The prefix set is closed: a well-formed but unknown prefix must fail."""
    with pytest.raises(ValueError, match="unknown identifier prefix"):
        mint(bad)


def test_every_known_prefix_can_be_minted():
    from or_audit.primitives import ENTITY_ID_PATTERN, ENTITY_PREFIXES

    for prefix in ENTITY_PREFIXES:
        assert re.fullmatch(ENTITY_ID_PATTERN, mint(prefix))


def test_model_rejects_id_with_wrong_prefix():
    """A surgeon id in the episode slot must fail validation, not silently pass."""
    with pytest.raises(ValidationError):
        _Holder(episode_id=new_surgeon_id(), surgeon_id=new_surgeon_id())


def test_model_accepts_correctly_prefixed_ids():
    holder = _Holder(episode_id=new_episode_id(), surgeon_id=new_surgeon_id())
    assert holder.episode_id.startswith("epi_")


def test_model_rejects_truncated_id():
    with pytest.raises(ValidationError):
        _Holder(episode_id="epi_TOOSHORT", surgeon_id=new_surgeon_id())
