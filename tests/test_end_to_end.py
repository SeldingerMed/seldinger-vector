"""End-to-end: does the whole thing actually work together?

The unit tests prove each layer holds its contract. This file proves the
contracts compose -- that an episode can enter as raw synthetic media and leave
as a credentialing report with an audit chain that verifies, without any layer
being bypassed on the way.

It also pins the properties that only exist at the seam: that scoring cannot be
reached without clearance, that attribution obeys section 9, and that the audit
chain covers the whole run rather than parts of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from or_audit.audit.trail import AuditAction
from or_audit.cli import main
from or_audit.demo import run_demo, synthetic_source
from or_audit.domain.enums import DeidStatus, Determination, GateStatus, MediaKind
from or_audit.errors import AuditChainError, DeidentificationBoundaryError
from or_audit.media.frames import NpzFrameSource


@pytest.fixture(scope="module")
def outcome(tmp_path_factory: pytest.TempPathFactory):
    """One demo run, shared across the module. Deterministic by construction."""
    workdir = tmp_path_factory.mktemp("demo")
    return run_demo(workdir, episodes=6), workdir


class TestTheChainComposes:
    def test_every_episode_produced_a_determination(self, outcome):
        result, _ = outcome
        assert len(result.assessments) == 6
        for assessment in result.assessments:
            assert assessment.decision.determination in set(Determination)

    def test_safety_gates_were_actually_evaluated(self, outcome):
        result, _ = outcome
        for assessment in result.assessments:
            assert len(assessment.gates) == 3
            assert assessment.gates.all_clear

    def test_the_critical_view_gate_passed_on_its_own_evidence(self, outcome):
        """Not a default pass: the reason must cite the timing it checked."""
        result, _ = outcome
        cvs = result.assessments[0].gates.results[0]
        assert cvs.status is GateStatus.PASS
        assert "before clipping began" in cvs.reason

    def test_determinations_track_the_scores(self, outcome):
        """First half scored 5/7 (below benchmark), second half 7/7."""
        result, _ = outcome
        determinations = [a.decision.determination for a in result.assessments]
        assert determinations[:3] == [Determination.DOES_NOT_MEET] * 3
        assert determinations[3:] == [Determination.MEETS_BENCHMARK] * 3

    def test_the_report_renders_with_every_episode(self, outcome):
        result, _ = outcome
        text = result.report.render()
        assert "CREDENTIALING REPORT" in text
        for assessment in result.assessments:
            assert assessment.episode.id in text

    def test_the_learning_curve_has_a_direction(self, outcome):
        result, _ = outcome
        assert "improving" in result.report.curve.trend()

    def test_determination_counts_add_up(self, outcome):
        result, _ = outcome
        assert sum(result.report.counts.values()) == len(result.assessments)


class TestDeidentificationActuallyHappened:
    """Section 8 at the seam: the pipeline must not have skipped it."""

    def test_media_reaching_scoring_was_attested(self, outcome):
        result, _ = outcome
        for assessment in result.assessments:
            assert assessment.episode.deid_status is DeidStatus.ATTESTED

    def test_out_of_body_frames_were_dropped_from_the_written_output(self, outcome):
        """Checks the file, not the plan."""
        _, workdir = outcome
        source = synthetic_source()
        redacted = NpzFrameSource(Path(workdir) / "case-1.npz")
        assert redacted.frame_count < source.frame_count

    def test_the_overlay_region_is_zeroed_in_the_written_output(self, outcome):
        _, workdir = outcome
        redacted = NpzFrameSource(Path(workdir) / "case-1.npz")
        assert redacted.read(0).pixels[0:16, 0:40].max() == 0

    def test_no_surviving_out_of_body_frame_reached_the_output(self, outcome):
        """Measured below the masked overlay band.

        Sampling the whole frame would be wrong: the redacted overlay is zeroed,
        which drags the mean down and makes in-body frames look like room
        frames. The rows below the overlay are unmasked, so in-body and
        out-of-body remain cleanly separable there.
        """
        _, workdir = outcome
        redacted = NpzFrameSource(Path(workdir) / "case-1.npz")
        leaked = [
            index
            for index in range(redacted.frame_count)
            if float(redacted.read(index).pixels[40:, :, 0].mean()) < 170.0
        ]
        assert not leaked, f"out-of-body frames survived at {leaked}"

    def test_perception_is_bound_to_the_cleared_media(self, outcome):
        result, _ = outcome
        for assessment in result.assessments:
            assert assessment.perception.media_sha256 == (assessment.episode.media[0].sha256,)
            assert assessment.perception.deid_attestation_sha256 == (
                assessment.episode.media[0].deid_attestation_sha256,
            )


class TestAuditChainCoversTheWholeRun:
    def test_the_chain_verifies_against_a_pinned_head_and_length(self, outcome):
        result, _ = outcome
        result.trail.verify(expected_head=result.trail.head_hash, expected_length=len(result.trail))

    def test_every_stage_appears_in_the_trail(self, outcome):
        result, _ = outcome
        actions = {entry.action for entry in result.trail}
        assert {
            AuditAction.EPISODE_REGISTERED,
            AuditAction.MEDIA_REGISTERED,
            AuditAction.DEID_STARTED,
            AuditAction.DEID_ATTESTED,
            AuditAction.PERCEPTION_COMPLETED,
            AuditAction.SAFETY_GATE_EVALUATED,
            AuditAction.SCORE_COMPUTED,
            AuditAction.DECISION_ISSUED,
        } <= actions

    def test_each_decision_entry_carries_the_record_digest(self, outcome):
        result, _ = outcome
        digests = {
            entry.payload["decision_sha256"]
            for entry in result.trail
            if entry.action is AuditAction.DECISION_ISSUED
        }
        assert digests == {a.decision.digest for a in result.assessments}

    def test_tampering_with_the_persisted_chain_is_detected(self, outcome, tmp_path):
        result, _ = outcome
        path = tmp_path / "audit.jsonl"
        result.trail.to_jsonl(path)
        text = path.read_text(encoding="utf-8").replace("meets_benchmark", "does_not_meet", 1)
        path.write_text(text, encoding="utf-8")
        with pytest.raises(AuditChainError):
            type(result.trail).from_jsonl(path)


class TestAttributionObeysSectionNine:
    """The demo institution's privilege posture is unconfirmed on purpose."""

    def test_individual_attribution_is_withheld(self, outcome):
        result, _ = outcome
        assert result.report.surgeon_ref is None
        for card in result.report.scorecards:
            assert card.surgeon_ref is None

    def test_the_report_explains_why(self, outcome):
        result, _ = outcome
        assert result.report.attribution_note is not None
        assert "peer-review protection" in result.report.attribution_note
        assert "attribution withheld" in result.report.render()

    def test_no_customer_reference_leaks_into_the_rendered_report(self, outcome):
        """The manifest carried DEMO-CASE-nnn; it must not appear."""
        result, _ = outcome
        assert "DEMO-CASE" not in result.report.render()


class TestScoringCannotBypassClearance:
    """The seam check, independent of the per-layer ones."""

    def test_an_episode_whose_attestation_is_not_held_is_refused(self, outcome):
        """require_cleared catches an asset pointing at a missing attestation."""
        from or_audit.pipeline import require_cleared

        result, _ = outcome
        episode = result.assessments[0].episode
        with pytest.raises(DeidentificationBoundaryError, match="cannot be evidenced"):
            require_cleared(episode, ())

    def test_uncleared_media_fails_the_read_gate_at_the_seam(self, outcome):
        result, _ = outcome
        episode = result.assessments[0].episode
        raw = episode.model_copy(
            update={
                "media": (
                    episode.media[0].model_copy(
                        update={
                            "deid_status": DeidStatus.RAW,
                            "deid_attestation_sha256": None,
                        }
                    ),
                )
            }
        )
        with pytest.raises(DeidentificationBoundaryError):
            raw.require_readable()


class TestCli:
    def test_demo_command_runs_and_reports(self, capsys, tmp_path):
        code = main(["demo", "--episodes", "2", "--workdir", str(tmp_path / "w")])
        out = capsys.readouterr().out
        assert code == 0
        assert "CREDENTIALING REPORT" in out
        assert "verification intact" in out
        assert "synthetic data" in out, "the disclaimer must always be printed"

    def test_demo_writes_a_verifiable_audit_log(self, capsys, tmp_path):
        log = tmp_path / "audit.jsonl"
        main(
            [
                "demo",
                "--episodes",
                "1",
                "--workdir",
                str(tmp_path / "w"),
                "--audit-log",
                str(log),
            ]
        )
        capsys.readouterr()
        assert main(["verify-audit", str(log), "--allow-unpinned"]) == 0
        assert "intact" in capsys.readouterr().out

    def test_verify_audit_warns_without_a_pinned_head(self, capsys, tmp_path):
        log = tmp_path / "audit.jsonl"
        main(["demo", "--episodes", "1", "--workdir", str(tmp_path / "w"), "--audit-log", str(log)])
        capsys.readouterr()
        main(["verify-audit", str(log), "--allow-unpinned"])
        assert "Tail truncation is not detectable" in capsys.readouterr().out

    def test_verify_audit_detects_truncation_with_a_pinned_head(self, capsys, tmp_path):
        log = tmp_path / "audit.jsonl"
        main(["demo", "--episodes", "1", "--workdir", str(tmp_path / "w"), "--audit-log", str(log)])
        capsys.readouterr()
        main(["verify-audit", str(log), "--allow-unpinned"])
        head = capsys.readouterr().out.split("head ")[1].split()[0]
        lines = log.read_text(encoding="utf-8").splitlines()
        log.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        assert main(["verify-audit", str(log), "--expected-head", head]) == 1
        assert "truncated from the end" in capsys.readouterr().err

    def test_verify_audit_rejects_a_missing_file(self, capsys, tmp_path):
        assert main(["verify-audit", str(tmp_path / "nope.jsonl")]) == 2

    def test_describe_rule_is_publishable(self, capsys):
        assert main(["describe-rule"]) == 0
        out = capsys.readouterr().out
        assert "DecisionRule 1" in out
        assert "85%" in out


class TestHardeningFromReview:
    """Follow-ups from the end-to-end review."""

    def test_unpinned_verification_does_not_exit_clean(self, capsys, tmp_path):
        """The pin's provenance is the security property.

        A zero exit from a check that could not detect truncation would be read
        by a script as a full pass.
        """
        log = tmp_path / "audit.jsonl"
        main(["demo", "--episodes", "1", "--workdir", str(tmp_path / "w"), "--audit-log", str(log)])
        capsys.readouterr()
        assert main(["verify-audit", str(log)]) == 3
        captured = capsys.readouterr()
        assert "INCOMPLETE" in captured.err

    def test_unpinned_verification_can_be_acknowledged_explicitly(self, capsys, tmp_path):
        log = tmp_path / "audit.jsonl"
        main(["demo", "--episodes", "1", "--workdir", str(tmp_path / "w"), "--audit-log", str(log)])
        capsys.readouterr()
        assert main(["verify-audit", str(log), "--allow-unpinned"]) == 0

    def test_a_pinned_verification_exits_clean(self, capsys, tmp_path):
        log = tmp_path / "audit.jsonl"
        main(["demo", "--episodes", "1", "--workdir", str(tmp_path / "w"), "--audit-log", str(log)])
        out = capsys.readouterr().out
        head = out.split("head         ")[1].split()[0]
        assert main(["verify-audit", str(log), "--expected-head", head]) == 0

    def test_the_trend_reports_its_sample_size(self, outcome):
        """A direction on two-versus-two episodes must not read as an effect size."""
        result, _ = outcome
        trend = result.report.curve.trend()
        assert "n=" in trend
        assert "not an effect size" in trend

    def test_coarse_policy_media_is_refused_downstream_not_merely_unattested(self):
        """The 'analyse but cannot attest' rule holds at the next stage too."""
        from or_audit.deid.pipeline import analyze
        from or_audit.deid.policy import DeidPolicy
        from or_audit.demo import synthetic_source
        from or_audit.domain.entities import MediaAsset
        from or_audit.domain.ids import new_episode_id, new_media_asset_id

        source = synthetic_source()
        asset = MediaAsset(
            id=new_media_asset_id(),
            episode_id=new_episode_id(),
            kind=MediaKind.ENDOSCOPIC_VIDEO,
            raw_uri="s3://raw/case.mp4",
            sha256="a" * 64,
            duration_seconds=9.0,
            deid_status=DeidStatus.RAW,
        )
        coarse = DeidPolicy(overlay_block_px=32, overlay_recall_justification="triage")
        analysed, _ = analyze(asset, source, coarse)
        assert analysed.deid_status is DeidStatus.IN_PROGRESS
        # Nothing past analysis accepts it: the read gate refuses IN_PROGRESS.
        with pytest.raises(DeidentificationBoundaryError):
            analysed.require_readable()
