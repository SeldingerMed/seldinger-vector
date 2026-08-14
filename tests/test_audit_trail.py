"""Audit-chain contract from PLAN.md section 7.3.

The chain's whole job is to make retroactive edits detectable, so most of
these tests mutate a valid chain and assert that verification catches it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from or_audit.audit.trail import (
    GENESIS_HASH,
    Actor,
    ActorKind,
    AuditAction,
    AuditTrail,
    verify_entries,
)
from or_audit.errors import AuditChainError

SUBJECT = "epi_0000000000000000000000000A"
OTHER_SUBJECT = "epi_0000000000000000000000000B"


@pytest.fixture
def trail(frozen_clock, actor) -> AuditTrail:
    built = AuditTrail(clock=frozen_clock)
    built.append(
        actor=actor,
        action=AuditAction.EPISODE_REGISTERED,
        subject_ref=SUBJECT,
        payload={"platform": "hugo"},
    )
    built.append(
        actor=actor,
        action=AuditAction.DEID_ATTESTED,
        subject_ref=SUBJECT,
        payload={"assets": 2},
    )
    return built


class TestChainConstruction:
    def test_empty_trail_head_is_genesis(self, frozen_clock):
        assert AuditTrail(clock=frozen_clock).head_hash == GENESIS_HASH

    def test_first_entry_links_to_genesis(self, trail):
        assert trail.entries[0].prev_hash == GENESIS_HASH

    def test_each_entry_links_to_its_predecessor(self, trail):
        assert trail.entries[1].prev_hash == trail.entries[0].entry_hash

    def test_sequence_is_dense_and_zero_based(self, trail):
        assert [e.seq for e in trail.entries] == [0, 1]

    def test_head_hash_tracks_last_entry(self, trail):
        assert trail.head_hash == trail.entries[-1].entry_hash

    def test_valid_chain_verifies(self, trail):
        trail.verify()

    def test_entries_carry_schema_version(self, trail):
        """Records must be self-describing, and the version must be hashed."""
        assert all(e.schema_version for e in trail.entries)

    def test_len_and_iteration(self, trail):
        assert len(trail) == 2
        assert [e.action for e in trail] == [
            AuditAction.EPISODE_REGISTERED,
            AuditAction.DEID_ATTESTED,
        ]

    def test_identical_payloads_produce_distinct_entry_hashes(self, frozen_clock, actor):
        """Sequence and prev_hash must disambiguate otherwise-identical events."""
        built = AuditTrail(clock=frozen_clock)
        for _ in range(2):
            built.append(
                actor=actor,
                action=AuditAction.SCORE_COMPUTED,
                subject_ref=SUBJECT,
                payload={"same": True},
            )
        assert built.entries[0].entry_hash != built.entries[1].entry_hash


class TestNoPhiAtTheBoundary:
    """Section 8: the append-only record must not be able to hold a name."""

    @pytest.mark.parametrize(
        "bad_ref",
        ["patient-mrn-98765", "Jane Smith", "epi_short", "", "EPI_0000000000000000000000000A"],
    )
    def test_free_text_subject_ref_rejected(self, frozen_clock, actor, bad_ref):
        built = AuditTrail(clock=frozen_clock)
        with pytest.raises(ValidationError):
            built.append(actor=actor, action=AuditAction.EXPORT_GRANTED, subject_ref=bad_ref)

    @pytest.mark.parametrize("bad_ref", ["Dr. Jane Smith MRN 12345", "Jane Smith", "A B", ""])
    def test_personal_name_actor_ref_rejected(self, bad_ref):
        with pytest.raises(ValidationError):
            Actor(kind=ActorKind.OPERATOR, ref=bad_ref)

    @pytest.mark.parametrize("good_ref", ["deid-pipeline", "rater.0041", "svc_scoring", "a"])
    def test_slug_actor_refs_accepted(self, good_ref):
        assert Actor(kind=ActorKind.RATER, ref=good_ref).ref == good_ref

    @pytest.mark.parametrize("bad_ref", ["12345678", "123-45-6789", "1", "000.11.2222"])
    def test_bare_identifier_numbers_rejected_as_actor_ref(self, bad_ref):
        """MRN- and SSN-shaped refs are the highest-risk PHI shapes."""
        with pytest.raises(ValidationError, match="bare identifier number"):
            Actor(kind=ActorKind.OPERATOR, ref=bad_ref)

    def test_alphanumeric_handle_with_digits_still_accepted(self):
        """The control targets purely numeric refs, not any ref containing digits."""
        assert Actor(kind=ActorKind.RATER, ref="rater-0041").ref == "rater-0041"

    @pytest.mark.parametrize(
        "bad_ref",
        [
            "mrn_AAAAAAAAAAAAAAAAAAAAAAAAAA",
            "pat_AAAAAAAAAAAAAAAAAAAAAAAAAA",
            "ep_AAAAAAAAAAAAAAAAAAAAAAAAAA",
        ],
    )
    def test_unknown_entity_prefix_rejected_as_subject(self, frozen_clock, actor, bad_ref):
        """The prefix set is closed, so a well-formed unknown prefix must fail."""
        built = AuditTrail(clock=frozen_clock)
        with pytest.raises(ValidationError):
            built.append(actor=actor, action=AuditAction.EXPORT_GRANTED, subject_ref=bad_ref)

    def test_payload_accepts_arbitrary_structured_content(self, frozen_clock, actor):
        """Documents an honest limitation rather than asserting a guarantee.

        Unlike ``subject_ref`` and ``actor.ref``, the payload is an arbitrary
        structured field and cannot be pattern-constrained without breaking
        its purpose. It is therefore the widest PHI channel into an
        append-only, exportable record, and keeping identifiers out of it is a
        caller obligation with no structural backstop. If a future phase adds
        a key allowlist, this test should start failing -- which is the point.
        """
        built = AuditTrail(clock=frozen_clock)
        built.append(
            actor=actor,
            action=AuditAction.SCORE_COMPUTED,
            subject_ref=SUBJECT,
            payload={"free": "anything at all", "nested": {"also": ["free"]}},
        )
        built.verify()
        assert built.entries[0].payload["free"] == "anything at all"


class TestTamperDetection:
    def test_payload_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[0] = tampered[0].model_copy(
            update={"payload_canonical": '{"platform":"da_vinci_5"}'}
        )
        with pytest.raises(AuditChainError, match="payload does not match"):
            verify_entries(tampered)

    def test_action_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[1] = tampered[1].model_copy(update={"action": AuditAction.DEID_FAILED})
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_subject_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[0] = tampered[0].model_copy(update={"subject_ref": OTHER_SUBJECT})
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_actor_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[0] = tampered[0].model_copy(
            update={"actor": Actor(kind=ActorKind.OPERATOR, ref="someone-else")}
        )
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_timestamp_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        moved = tampered[0].recorded_at.replace(year=2030)
        tampered[0] = tampered[0].model_copy(update={"recorded_at": moved})
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_schema_version_edit_is_detected(self, trail):
        tampered = list(trail.entries)
        tampered[0] = tampered[0].model_copy(update={"schema_version": "99"})
        with pytest.raises(AuditChainError, match="hash does not match"):
            verify_entries(tampered)

    def test_deleting_a_middle_entry_is_detected(self, trail):
        with pytest.raises(AuditChainError, match="declares seq"):
            verify_entries(list(trail.entries)[1:])

    def test_reordering_entries_is_detected(self, trail):
        with pytest.raises(AuditChainError, match="declares seq"):
            verify_entries(list(reversed(trail.entries)))

    def test_broken_link_is_detected(self, trail):
        tampered = list(trail.entries)
        forged = tampered[1].model_copy(update={"prev_hash": GENESIS_HASH})
        # Re-derive the entry hash so only the link itself is inconsistent.
        _, entry_hash = forged.recompute_hashes()
        tampered[1] = forged.model_copy(update={"entry_hash": entry_hash})
        with pytest.raises(AuditChainError, match="prev_hash does not match"):
            verify_entries(tampered)

    def test_foreign_chain_version_is_rejected(self, trail):
        tampered = list(trail.entries)
        forged = tampered[0].model_copy(update={"chain_version": "99"})
        _, entry_hash = forged.recompute_hashes()
        tampered[0] = forged.model_copy(update={"entry_hash": entry_hash})
        with pytest.raises(AuditChainError, match="chain version"):
            verify_entries(tampered)

    def test_naive_timestamp_raises_chain_error_not_value_error(self, trail):
        """Error taxonomy: an unverifiable entry surfaces as AuditChainError.

        ``model_copy`` bypasses validators, so this is the path a corrupted
        record actually takes. Leaking the serializer's ValueError would give
        callers two unrelated exception types to catch for one condition.
        """
        naive = trail.entries[0].recorded_at.replace(tzinfo=None)
        tampered = list(trail.entries)
        tampered[0] = tampered[0].model_copy(update={"recorded_at": naive})
        with pytest.raises(AuditChainError, match="cannot be canonicalized"):
            verify_entries(tampered)

    def test_naive_timestamp_rejected_at_construction(self, trail):
        """Deserialization still catches it earlier, with a clearer message."""
        payload = trail.entries[0].model_dump(mode="python")
        payload["recorded_at"] = trail.entries[0].recorded_at.replace(tzinfo=None)
        with pytest.raises(AuditChainError, match="timezone-aware"):
            type(trail.entries[0]).model_validate(payload)


class TestTailTruncation:
    """Dropping the newest entries needs no rewriting, so the chain alone
    cannot catch it. Verification must accept an externally pinned head."""

    def test_truncated_tail_verifies_without_a_pin(self, trail):
        verify_entries(list(trail.entries)[:1])

    def test_truncated_tail_is_caught_by_pinned_head(self, trail):
        pinned = trail.head_hash
        with pytest.raises(AuditChainError, match="truncated from the end"):
            verify_entries(list(trail.entries)[:1], expected_head=pinned)

    def test_truncated_tail_is_caught_by_pinned_length(self, trail):
        with pytest.raises(AuditChainError, match="expected 2"):
            verify_entries(list(trail.entries)[:1], expected_length=2)

    def test_fully_emptied_chain_is_caught_by_pinned_head(self, trail):
        with pytest.raises(AuditChainError, match="empty chain"):
            verify_entries([], expected_head=trail.head_hash)

    def test_intact_chain_passes_its_pins(self, trail):
        trail.verify(expected_head=trail.head_hash, expected_length=2)


class TestPersistence:
    def test_jsonl_roundtrip_preserves_and_verifies(self, trail, tmp_path):
        path = tmp_path / "audit.jsonl"
        trail.to_jsonl(path)
        loaded = AuditTrail.from_jsonl(path)
        assert [e.entry_hash for e in loaded] == [e.entry_hash for e in trail]
        assert loaded.head_hash == trail.head_hash
        assert [e.payload for e in loaded] == [e.payload for e in trail]

    def test_loading_a_tampered_file_raises(self, trail, tmp_path):
        path = tmp_path / "audit.jsonl"
        trail.to_jsonl(path)
        text = path.read_text(encoding="utf-8").replace("hugo", "dvci")
        path.write_text(text, encoding="utf-8")
        with pytest.raises(AuditChainError):
            AuditTrail.from_jsonl(path)

    def test_tampered_file_loadable_with_verification_disabled(self, trail, tmp_path):
        """Inspecting a known-broken log must still be possible."""
        path = tmp_path / "audit.jsonl"
        trail.to_jsonl(path)
        text = path.read_text(encoding="utf-8").replace("hugo", "dvci")
        path.write_text(text, encoding="utf-8")
        loaded = AuditTrail.from_jsonl(path, verify=False)
        assert len(loaded) == 2
        with pytest.raises(AuditChainError):
            loaded.verify()

    def test_blank_lines_are_tolerated(self, trail, tmp_path):
        path = tmp_path / "audit.jsonl"
        trail.to_jsonl(path)
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(AuditTrail.from_jsonl(path)) == 2


class TestPayloadHandling:
    def test_absent_payload_becomes_empty_mapping(self, frozen_clock, actor):
        built = AuditTrail(clock=frozen_clock)
        entry = built.append(actor=actor, action=AuditAction.EXPORT_DENIED, subject_ref=SUBJECT)
        assert entry.payload == {}
        built.verify()

    def test_top_level_mutation_after_append_is_harmless(self, frozen_clock, actor):
        built = AuditTrail(clock=frozen_clock)
        payload = {"n": 1}
        built.append(
            actor=actor, action=AuditAction.SCORE_COMPUTED, subject_ref=SUBJECT, payload=payload
        )
        payload["n"] = 2
        built.verify()
        assert built.entries[0].payload == {"n": 1}

    def test_nested_mutation_after_append_is_harmless(self, frozen_clock, actor):
        """A shallow copy would let this silently invalidate a valid chain."""
        built = AuditTrail(clock=frozen_clock)
        payload = {"scores": {"gears": 3}}
        built.append(
            actor=actor, action=AuditAction.SCORE_COMPUTED, subject_ref=SUBJECT, payload=payload
        )
        payload["scores"]["gears"] = 5
        built.verify()
        assert built.entries[0].payload == {"scores": {"gears": 3}}

    def test_mutating_the_returned_payload_is_harmless(self, frozen_clock, actor):
        """The accessor must not hand out a reference into the record."""
        built = AuditTrail(clock=frozen_clock)
        built.append(
            actor=actor,
            action=AuditAction.SCORE_COMPUTED,
            subject_ref=SUBJECT,
            payload={"a": 1},
        )
        built.entries[0].payload["a"] = 99
        built.verify()
        assert built.entries[0].payload == {"a": 1}

    def test_uncanonicalizable_payload_is_rejected(self, frozen_clock, actor):
        built = AuditTrail(clock=frozen_clock)
        with pytest.raises(ValueError, match="non-finite"):
            built.append(
                actor=actor,
                action=AuditAction.SCORE_COMPUTED,
                subject_ref=SUBJECT,
                payload={"score": float("nan")},
            )
        assert len(built) == 0, "a rejected append must not leave a partial entry"
