"""The Phase 1 agreement gate, PLAN.md section 13.

Each test corresponds to one of the three defects in the plan's v1 gate:
unspecified ICC form, an absolute target, and an unstratified cohort. Plus the
endpoint ordering, which the plan states and which is easy to quietly invert
by shipping GEARS alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from or_audit.domain.enums import SkillBand
from or_audit.errors import ScoreContractError
from or_audit.metrics.harness import (
    AgreementFigure,
    AgreementGate,
    Endpoint,
    agreement_figure,
)

RNG = np.random.default_rng(20260304)


def cohort(n: int, *, noise: float, raters: int = 3, spread: float = 1.0):
    """Build a synthetic cohort with a known agreement structure.

    ``spread`` controls between-case variance and ``noise`` controls rater
    disagreement, so the two axes the ICC is meant to separate can be varied
    independently.
    """
    truth = RNG.normal(0.0, spread, n)
    panel = [truth + RNG.normal(0.0, noise, n) for _ in range(raters)]
    return truth, panel


class TestRelativeTargetNotAbsolute:
    """The panel is the ceiling; an absolute target ignores it."""

    def test_scorer_matching_a_tight_panel_passes(self):
        truth, panel = cohort(60, noise=0.15)
        automated = truth + RNG.normal(0.0, 0.15, 60)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=automated,
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        assert figure.passes
        assert figure.required == pytest.approx(0.9 * figure.expert_vs_expert.value)

    def test_target_scales_down_when_the_panel_disagrees(self):
        """With a noisy panel the bar drops, which is the point.

        An absolute 0.8 would fail a scorer that matches its experts as well as
        they match each other -- penalising the model for the rubric's noise.
        """
        truth, panel = cohort(60, noise=1.2)
        automated = truth + RNG.normal(0.0, 1.2, 60)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=automated,
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        assert figure.expert_vs_expert.value < 0.8
        assert figure.passes

    def test_a_poor_scorer_still_fails_against_a_noisy_panel(self):
        """Scaling the bar must not remove it."""
        _truth, panel = cohort(60, noise=0.8)
        automated = RNG.normal(0.0, 1.0, 60)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=automated,
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        assert not figure.passes

    def test_panel_agreement_is_always_reported_alongside(self):
        truth, panel = cohort(40, noise=0.3)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=truth,
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        assert "vs panel" in figure.describe()
        assert f"{figure.expert_vs_expert.value:.3f}" in figure.describe()


class TestPanelMustBeMeasurable:
    def test_a_single_expert_gives_no_ceiling(self):
        truth, panel = cohort(40, noise=0.3, raters=1)
        with pytest.raises(ScoreContractError, match="at least 2 expert raters"):
            agreement_figure(
                endpoint=Endpoint.BINARY_PROFICIENCY,
                automated=truth,
                expert_panel=panel,
                band=SkillBand.ATTENDING,
            )

    def test_raters_must_score_the_same_cases(self):
        truth, panel = cohort(40, noise=0.3)
        panel[0] = panel[0][:20]
        with pytest.raises(ScoreContractError, match="exactly the same cases"):
            agreement_figure(
                endpoint=Endpoint.BINARY_PROFICIENCY,
                automated=truth,
                expert_panel=panel,
                band=SkillBand.ATTENDING,
            )

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_invalid_relative_target_is_refused(self, bad):
        truth, panel = cohort(40, noise=0.3)
        with pytest.raises(ScoreContractError, match="relative_target"):
            agreement_figure(
                endpoint=Endpoint.BINARY_PROFICIENCY,
                automated=truth,
                expert_panel=panel,
                band=SkillBand.ATTENDING,
                relative_target=bad,
            )


class TestStratification:
    """Mixed-band figures are inflated by between-group variance."""

    def test_mixed_band_figure_cannot_pass_the_gate(self):
        truth, panel = cohort(60, noise=0.15)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=truth + RNG.normal(0.0, 0.15, 60),
            expert_panel=panel,
            band=None,
        )
        assert figure.passes, "the raw figure is strong"
        verdict = AgreementGate().evaluate({Endpoint.BINARY_PROFICIENCY: figure})
        assert not verdict.passed
        assert "mixed-band" in verdict.reason

    def test_mixed_band_figure_is_labelled_in_its_description(self):
        truth, panel = cohort(40, noise=0.2)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=truth,
            expert_panel=panel,
            band=None,
        )
        assert "MIXED BAND" in figure.describe()
        assert not figure.is_stratified

    def test_within_band_figure_names_the_band(self):
        truth, panel = cohort(40, noise=0.2)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=truth,
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        assert "attending only" in figure.describe()


class TestEndpointOrdering:
    """Binary proficiency is primary; GEARS alone is not a result."""

    def _figure(self, endpoint: Endpoint, n: int = 60) -> AgreementFigure:
        truth, panel = cohort(n, noise=0.15)
        return agreement_figure(
            endpoint=endpoint,
            automated=truth + RNG.normal(0.0, 0.15, n),
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )

    def test_gears_alone_does_not_pass(self):
        verdict = AgreementGate().evaluate({Endpoint.GEARS: self._figure(Endpoint.GEARS)})
        assert not verdict.passed
        assert "primary endpoint" in verdict.reason

    def test_binary_proficiency_alone_can_pass(self):
        verdict = AgreementGate().evaluate(
            {Endpoint.BINARY_PROFICIENCY: self._figure(Endpoint.BINARY_PROFICIENCY)}
        )
        assert verdict.passed

    def test_gears_is_carried_as_secondary(self):
        verdict = AgreementGate().evaluate(
            {
                Endpoint.BINARY_PROFICIENCY: self._figure(Endpoint.BINARY_PROFICIENCY),
                Endpoint.GEARS: self._figure(Endpoint.GEARS),
            }
        )
        assert verdict.passed
        assert len(verdict.secondary) == 1
        assert "gears" in verdict.secondary[0]

    def test_a_failing_primary_is_not_rescued_by_a_strong_secondary(self):
        _truth, panel = cohort(60, noise=0.2)
        weak = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=RNG.normal(0.0, 1.0, 60),
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        verdict = AgreementGate().evaluate(
            {
                Endpoint.BINARY_PROFICIENCY: weak,
                Endpoint.GEARS: self._figure(Endpoint.GEARS),
            }
        )
        assert not verdict.passed


class TestCohortSize:
    def test_a_small_cohort_cannot_pass(self):
        truth, panel = cohort(10, noise=0.15)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=truth + RNG.normal(0.0, 0.15, 10),
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        verdict = AgreementGate().evaluate({Endpoint.BINARY_PROFICIENCY: figure})
        assert not verdict.passed
        assert "case floor" in verdict.reason

    def test_the_floor_is_configurable(self):
        truth, panel = cohort(10, noise=0.15)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=truth + RNG.normal(0.0, 0.15, 10),
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        assert AgreementGate(min_cases=5).evaluate({Endpoint.BINARY_PROFICIENCY: figure}).passed


class TestVerdictCannotBeCollapsed:
    def test_bool_raises_so_the_verdict_is_read_by_name(self):
        truth, panel = cohort(60, noise=0.15)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=truth,
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        verdict = AgreementGate().evaluate({Endpoint.BINARY_PROFICIENCY: figure})
        with pytest.raises(ScoreContractError, match=r"read AgreementGateResult\.passed"):
            bool(verdict)

    def test_passed_is_readable(self):
        truth, panel = cohort(60, noise=0.15)
        figure = agreement_figure(
            endpoint=Endpoint.BINARY_PROFICIENCY,
            automated=truth,
            expert_panel=panel,
            band=SkillBand.ATTENDING,
        )
        assert AgreementGate().evaluate({Endpoint.BINARY_PROFICIENCY: figure}).passed is True
