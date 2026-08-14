"""The decision rule and contestation machinery, PLAN.md sections 7.2 and 7.3."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from or_audit.decision.record import (
    Contestation,
    ContestationState,
    DecisionRecord,
    RaterDisagreement,
    SubjectResponse,
    open_contestations,
)
from or_audit.decision.rule import DecisionRule, forbid_scalar_collapse
from or_audit.domain.enums import Determination, GateStatus, SkillBand, ThresholdOwner
from or_audit.domain.ids import new_decision_id, new_episode_id, new_surgeon_id
from or_audit.errors import DomainInvariantError, ScoreContractError
from or_audit.scoring.gates import GateId, GateResult, SafetyGateSet
from or_audit.scoring.skill import (
    ProficiencyItem,
    ProficiencyResult,
    ScoreVector,
    SkillScore,
)

DECIDED = datetime(2026, 3, 5, 9, 0, tzinfo=UTC)
LATER = DECIDED + timedelta(days=3)
ITEMS = list(ProficiencyItem)


def rule(**overrides: object) -> DecisionRule:
    base: dict[str, object] = {
        "version": "1",
        "threshold_owner": ThresholdOwner.CUSTOMER,
        "threshold_provenance": "Credentialing committee minute 2026-02-11",
    }
    return DecisionRule(**(base | overrides))


def gate(gate_id: GateId, status: GateStatus) -> GateResult:
    return GateResult(gate=gate_id, status=status, reason="r", perception="b@1")


def vector(*, gates: tuple[GateResult, ...] | None = None, met: int = 7) -> ScoreVector:
    results = gates or (
        gate(GateId.CVS, GateStatus.PASS),
        gate(GateId.BLEEDING, GateStatus.PASS),
    )
    proficiency = tuple(
        ProficiencyResult(item=item, met=index < met) for index, item in enumerate(ITEMS)
    )
    return ScoreVector(
        gates=SafetyGateSet(results=results),
        skill=SkillScore(
            band_at_episode=SkillBand.ATTENDING,
            rater="rater-0041",
            proficiency=proficiency,
        ),
    )


class TestHardGatesAreDispositive:
    """Section 7.1: a safety failure does not trade off against skill."""

    def test_failed_gate_forces_does_not_meet(self):
        gates = (gate(GateId.CVS, GateStatus.FAIL), gate(GateId.BLEEDING, GateStatus.PASS))
        determination, reason = rule().apply(vector(gates=gates, met=7))
        assert determination is Determination.DOES_NOT_MEET
        assert "dispositive" in reason

    def test_a_perfect_skill_score_does_not_offset_a_failed_gate(self):
        """This is precisely the averaging section 7.1 prohibits."""
        gates = (gate(GateId.CVS, GateStatus.FAIL),)
        determination, reason = rule().apply(vector(gates=gates, met=len(ITEMS)))
        assert determination is Determination.DOES_NOT_MEET
        assert "no skill score offsets it" in reason

    def test_failed_gate_outranks_an_unassessable_one(self):
        gates = (
            gate(GateId.CVS, GateStatus.FAIL),
            gate(GateId.BLEEDING, GateStatus.NOT_ASSESSABLE),
        )
        assert rule().apply(vector(gates=gates))[0] is Determination.DOES_NOT_MEET


class TestAbstention:
    """Section 7.2: the scorer must be able to decline to decide."""

    def test_unassessable_gate_yields_indeterminate_not_a_failure(self):
        """Missing evidence is not a finding against the surgeon."""
        gates = (gate(GateId.CVS, GateStatus.NOT_ASSESSABLE),)
        determination, reason = rule().apply(vector(gates=gates))
        assert determination is Determination.INDETERMINATE
        assert "not a finding against the surgeon" in reason

    def test_too_few_assessable_items_yields_indeterminate(self):
        proficiency = (
            ProficiencyResult(item=ITEMS[0], met=True),
            ProficiencyResult(item=ITEMS[1], met=None),
            ProficiencyResult(item=ITEMS[2], met=None),
        )
        thin = ScoreVector(
            gates=SafetyGateSet(results=(gate(GateId.CVS, GateStatus.PASS),)),
            skill=SkillScore(
                band_at_episode=SkillBand.ATTENDING,
                rater="rater-0041",
                proficiency=proficiency,
            ),
        )
        determination, reason = rule().apply(thin)
        assert determination is Determination.INDETERMINATE
        assert "carry meaning" in reason

    def test_unassessable_gates_can_be_configured_not_to_block(self):
        gates = (gate(GateId.CVS, GateStatus.NOT_ASSESSABLE),)
        permissive = rule(unassessable_gate_blocks=False)
        assert permissive.apply(vector(gates=gates))[0] is Determination.MEETS_BENCHMARK


class TestBenchmark:
    def test_meeting_the_benchmark_passes(self):
        determination, reason = rule().apply(vector(met=7))
        assert determination is Determination.MEETS_BENCHMARK
        assert "benchmark set by customer" in reason

    def test_falling_short_does_not_meet(self):
        determination, reason = rule().apply(vector(met=4))
        assert determination is Determination.DOES_NOT_MEET
        assert "below the 85% benchmark" in reason

    def test_the_boundary_is_inclusive(self):
        strict = rule(min_proficiency_fraction=1.0)
        assert strict.apply(vector(met=len(ITEMS)))[0] is Determination.MEETS_BENCHMARK

    def test_not_applicable_gates_are_noted_not_hidden(self):
        gates = (
            gate(GateId.CVS, GateStatus.NOT_APPLICABLE),
            gate(GateId.BLEEDING, GateStatus.PASS),
        )
        _, reason = rule().apply(vector(gates=gates))
        assert "not applicable" in reason

    def test_every_reason_names_the_rule_version(self):
        """A reader must never have to guess which rule applied."""
        for met in (0, 4, 7):
            assert "rule 1" in rule().apply(vector(met=met))[1]


class TestRuleProvenance:
    def test_a_society_threshold_must_cite_the_society(self):
        with pytest.raises(DomainInvariantError, match="must cite it"):
            rule(
                threshold_owner=ThresholdOwner.SPECIALTY_SOCIETY,
                threshold_provenance="agreed internally",
            )

    def test_a_cited_society_threshold_is_accepted(self):
        assert rule(
            threshold_owner=ThresholdOwner.SPECIALTY_SOCIETY,
            threshold_provenance="SAGES consensus statement 2026",
        )

    def test_describe_is_publishable_before_a_pilot(self):
        text = rule().describe()
        assert "DecisionRule 1" in text
        assert "85%" in text
        assert "customer" in text

    def test_there_is_no_scalar_score(self):
        with pytest.raises(ScoreContractError, match="no scalar score"):
            forbid_scalar_collapse(vector())


def record(**overrides: object) -> DecisionRecord:
    base: dict[str, object] = {
        "id": new_decision_id(),
        "episode_id": new_episode_id(),
        "surgeon_id": new_surgeon_id(),
        "determination": Determination.DOES_NOT_MEET,
        "reason": "rule 1: below benchmark",
        "decided_at": DECIDED,
        "decided_by": "svc-decision",
        "rule_version": "1",
        "perception_identity": "expert-annotation@1",
        "gate_policy_version": "1",
    }
    return DecisionRecord(**(base | overrides))


class TestRaterDisagreementIsSurfaced:
    """Section 7.3: where the panel split, the artifact says so."""

    def test_disagreement_appears_in_the_subject_disclosure(self):
        split = RaterDisagreement(
            subject="hepatocystic triangle cleared", positions=("yes", "no", "yes")
        )
        disclosure = record(disagreements=(split,)).subject_disclosure()
        assert disclosure["panel_disagreements"] == [
            {"subject": "hepatocystic triangle cleared", "positions": ["yes", "no", "yes"]}
        ]

    def test_a_unanimous_panel_is_distinguishable_from_an_unexamined_one(self):
        assert record().is_unanimous

    def test_a_unanimous_split_cannot_be_recorded_as_disagreement(self):
        with pytest.raises(DomainInvariantError, match="unanimous"):
            RaterDisagreement(subject="x", positions=("yes", "yes"))

    def test_a_single_position_is_not_a_disagreement(self):
        with pytest.raises(DomainInvariantError, match="at least two positions"):
            RaterDisagreement(subject="x", positions=("yes",))


class TestContestation:
    def _filed(self, **kw: object) -> Contestation:
        base: dict[str, object] = {
            "state": ContestationState.FILED,
            "filed_at": LATER,
            "filed_by": "sur-0041",
            "grounds": "the clipping phase was mislabelled",
        }
        return Contestation(**(base | kw))

    def test_filing_produces_a_new_record(self):
        """Records are immutable; the sequence of states stays inspectable."""
        original = record()
        challenged = original.with_contestation(self._filed())
        assert not original.contestations
        assert challenged.has_open_contestation

    def test_a_successful_challenge_supersedes_without_erasing(self):
        upheld = self._filed(
            state=ContestationState.UPHELD_FOR_SUBJECT,
            resolved_at=LATER + timedelta(days=1),
            revised_determination=Determination.MEETS_BENCHMARK,
        )
        revised = record().with_contestation(upheld)
        assert revised.effective_determination is Determination.MEETS_BENCHMARK
        assert revised.determination is Determination.DOES_NOT_MEET
        assert revised.was_revised
        assert not revised.is_adverse

    def test_an_unsuccessful_challenge_leaves_the_determination(self):
        stands = self._filed(
            state=ContestationState.ORIGINAL_STANDS, resolved_at=LATER + timedelta(days=1)
        )
        challenged = record().with_contestation(stands)
        assert challenged.effective_determination is Determination.DOES_NOT_MEET
        assert not challenged.was_revised
        assert challenged.is_adverse

    def test_upheld_must_name_the_revised_determination(self):
        with pytest.raises(DomainInvariantError, match="must name the revised"):
            self._filed(
                state=ContestationState.UPHELD_FOR_SUBJECT,
                resolved_at=LATER + timedelta(days=1),
            )

    def test_only_an_upheld_challenge_may_change_the_outcome(self):
        with pytest.raises(DomainInvariantError, match="only an upheld challenge"):
            self._filed(
                state=ContestationState.ORIGINAL_STANDS,
                resolved_at=LATER + timedelta(days=1),
                revised_determination=Determination.MEETS_BENCHMARK,
            )

    def test_a_resolved_challenge_must_say_when(self):
        with pytest.raises(DomainInvariantError, match="must record when it resolved"):
            self._filed(state=ContestationState.ORIGINAL_STANDS)

    def test_an_open_challenge_cannot_claim_a_resolution_time(self):
        with pytest.raises(DomainInvariantError, match="not resolved but names"):
            self._filed(state=ContestationState.UNDER_REVIEW, resolved_at=LATER)

    def test_a_challenge_cannot_resolve_before_it_was_filed(self):
        with pytest.raises(DomainInvariantError, match="cannot resolve before"):
            self._filed(state=ContestationState.ORIGINAL_STANDS, resolved_at=DECIDED)

    def test_a_challenge_cannot_predate_the_decision(self):
        with pytest.raises(DomainInvariantError, match="cannot be filed before"):
            record(contestations=(self._filed(filed_at=DECIDED - timedelta(days=1)),))

    def test_two_open_challenges_are_refused(self):
        """Concurrent challenges cannot be resolved coherently."""
        with pytest.raises(DomainInvariantError, match="only one contestation may be open"):
            record(contestations=(self._filed(), self._filed()))

    def test_a_withdrawn_challenge_reopens_the_appeal_path(self):
        withdrawn = self._filed(
            state=ContestationState.WITHDRAWN, resolved_at=LATER + timedelta(days=1)
        )
        challenged = record().with_contestation(withdrawn)
        assert not challenged.has_open_contestation
        assert challenged.subject_disclosure()["appeal_available"] is True

    def test_review_queue_finds_open_challenges(self):
        quiet = record()
        loud = record().with_contestation(self._filed())
        assert open_contestations([quiet, loud]) == (loud,)


class TestRightOfResponse:
    def test_a_response_is_attached_durably(self):
        response = SubjectResponse(
            submitted_at=LATER,
            submitted_by="sur-0041",
            statement="The anatomy was aberrant; see the operative note.",
        )
        answered = record().with_response(response)
        assert len(answered.responses) == 1
        assert "aberrant" in str(answered.subject_disclosure()["responses"])

    def test_a_response_cannot_predate_the_decision(self):
        early = SubjectResponse(
            submitted_at=DECIDED - timedelta(hours=1),
            submitted_by="sur-0041",
            statement="x",
        )
        with pytest.raises(DomainInvariantError, match="cannot predate"):
            record(responses=(early,))

    def test_naive_timestamps_are_refused(self):
        with pytest.raises(DomainInvariantError, match="timezone-aware"):
            SubjectResponse(
                submitted_at=datetime(2026, 3, 6, 9, 0),
                submitted_by="sur-0041",
                statement="x",
            )


class TestProvenanceIsPinned:
    """Section 7.3: the record must outlive the versions that produced it."""

    def test_versions_are_recorded_on_the_decision(self):
        disclosure = record().subject_disclosure()
        assert disclosure["rule_version"] == "1"
        assert disclosure["perception"] == "expert-annotation@1"
        assert disclosure["gate_policy_version"] == "1"

    def test_digest_changes_when_anything_changes(self):
        first = record()
        second = first.model_copy(update={"reason": "rule 1: something else"})
        assert first.digest != second.digest

    def test_appending_a_response_changes_the_digest(self):
        first = record()
        second = first.with_response(
            SubjectResponse(submitted_at=LATER, submitted_by="sur-0041", statement="x")
        )
        assert first.digest != second.digest

    def test_naive_decision_time_is_refused(self):
        with pytest.raises(DomainInvariantError, match="timezone-aware"):
            record(decided_at=datetime(2026, 3, 5, 9, 0))
