"""Soft skill scores and the vector that refuses to become a number."""

from __future__ import annotations

import pytest

from or_audit.domain.enums import GateStatus, SkillBand
from or_audit.errors import DomainInvariantError, ScoreContractError
from or_audit.scoring.gates import GateId, GateResult, SafetyGateSet
from or_audit.scoring.skill import (
    GearsDomain,
    GearsRating,
    ProficiencyItem,
    ProficiencyResult,
    ScoreVector,
    SkillScore,
)

ITEMS = list(ProficiencyItem)


def skill(**overrides: object) -> SkillScore:
    base: dict[str, object] = {
        "band_at_episode": SkillBand.ATTENDING,
        "rater": "rater-0041",
        "proficiency": tuple(ProficiencyResult(item=i, met=True) for i in ITEMS),
    }
    return SkillScore(**(base | overrides))


def gates(status: GateStatus = GateStatus.PASS) -> SafetyGateSet:
    return SafetyGateSet(
        results=(GateResult(gate=GateId.CVS, status=status, reason="r", perception="b@1"),)
    )


class TestPrimaryEndpointIsRequired:
    """Section 13: GEARS alone is not a result."""

    def test_gears_without_proficiency_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="GEARS alone is not a result"):
            SkillScore(
                band_at_episode=SkillBand.ATTENDING,
                rater="rater-0041",
                proficiency=(),
                gears=tuple(GearsRating(domain=d, score=4) for d in GearsDomain),
            )

    def test_proficiency_without_gears_is_fine(self):
        assert skill().gears == ()

    def test_duplicate_proficiency_item_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="proficiency item was scored twice"):
            skill(
                proficiency=(
                    ProficiencyResult(item=ITEMS[0], met=True),
                    ProficiencyResult(item=ITEMS[0], met=False),
                )
            )

    def test_duplicate_gears_domain_is_rejected(self):
        with pytest.raises(DomainInvariantError, match="GEARS domain was scored twice"):
            skill(
                gears=(
                    GearsRating(domain=GearsDomain.EFFICIENCY, score=3),
                    GearsRating(domain=GearsDomain.EFFICIENCY, score=5),
                )
            )


class TestProficiencyFraction:
    def test_all_met_is_one(self):
        assert skill().proficiency_fraction == pytest.approx(1.0)

    def test_half_met(self):
        results = tuple(
            ProficiencyResult(item=item, met=index % 2 == 0) for index, item in enumerate(ITEMS[:4])
        )
        assert skill(proficiency=results).proficiency_fraction == pytest.approx(0.5)

    def test_unassessable_items_leave_the_denominator(self):
        """Poor video must not read as poor performance."""
        results = (
            ProficiencyResult(item=ITEMS[0], met=True),
            ProficiencyResult(item=ITEMS[1], met=None),
            ProficiencyResult(item=ITEMS[2], met=None),
        )
        score = skill(proficiency=results)
        assert score.proficiency_fraction == pytest.approx(1.0)
        assert len(score.unassessable) == 2
        assert len(score.assessable) == 1

    def test_nothing_assessable_is_undefined_not_zero(self):
        """A zero denominator is not a score of zero; it is missing evidence."""
        results = tuple(ProficiencyResult(item=i, met=None) for i in ITEMS[:3])
        with pytest.raises(ScoreContractError, match="undefined"):
            _ = skill(proficiency=results).proficiency_fraction

    def test_met_count_ignores_unassessable(self):
        results = (
            ProficiencyResult(item=ITEMS[0], met=True),
            ProficiencyResult(item=ITEMS[1], met=None),
            ProficiencyResult(item=ITEMS[2], met=False),
        )
        assert skill(proficiency=results).met_count == 1


class TestGearsIsSecondary:
    def test_complete_gears_totals(self):
        score = skill(gears=tuple(GearsRating(domain=d, score=4) for d in GearsDomain))
        assert score.gears_total == 24

    def test_partial_gears_has_no_total(self):
        """A partial total is not comparable to a complete one."""
        score = skill(gears=(GearsRating(domain=GearsDomain.EFFICIENCY, score=4),))
        assert score.gears_total is None

    @pytest.mark.parametrize("bad", [0, 6, -1])
    def test_scores_outside_one_to_five_are_rejected(self, bad):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GearsRating(domain=GearsDomain.EFFICIENCY, score=bad)


class TestScoreVectorRefusesToCollapse:
    """Section 7.1: hard gates never average into soft scores."""

    def test_float_raises(self):
        vector = ScoreVector(gates=gates(), skill=skill())
        with pytest.raises(ScoreContractError, match="no scalar value"):
            float(vector)

    def test_int_raises(self):
        vector = ScoreVector(gates=gates(), skill=skill())
        with pytest.raises(ScoreContractError, match="no scalar value"):
            int(vector)

    def test_bool_raises(self):
        vector = ScoreVector(gates=gates(), skill=skill())
        with pytest.raises(ScoreContractError, match="no truth value"):
            bool(vector)

    def test_error_points_at_the_decision_rule(self):
        """A caller who wants one answer must go somewhere accountable."""
        vector = ScoreVector(gates=gates(), skill=skill())
        with pytest.raises(ScoreContractError, match="decision rule"):
            float(vector)

    def test_components_remain_separately_readable(self):
        vector = ScoreVector(gates=gates(), skill=skill())
        assert vector.gates.all_clear
        assert vector.skill.proficiency_fraction == pytest.approx(1.0)

    def test_a_failing_gate_does_not_change_the_skill_score(self):
        """The two are different kinds of judgement and must not interact."""
        vector = ScoreVector(gates=gates(GateStatus.FAIL), skill=skill())
        assert vector.gates.any_failed
        assert vector.skill.proficiency_fraction == pytest.approx(1.0)
