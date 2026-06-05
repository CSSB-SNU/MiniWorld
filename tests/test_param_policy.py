"""Tests for ``miniworld.training.param_policy``.

Covers pattern matching, conflict detection, default fallback,
re-initialization (project Linear + LayerNorm + vanilla torch Linear),
checkpoint loading with shape/missing fallback to reinit, and the
trainable_parameters helper.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from miniworld.training.param_policy import (
    ParamPolicyConfig,
    ParamPolicyConfigError,
    ParamPolicyConflictError,
    _classify_one,
    _matches,
    apply_param_policy,
    classify_params,
    format_summary,
    trainable_parameters,
    validate_policy,
)


# Tiny model that mixes project Linear, project LayerNorm, and a generic
# module, so we can exercise every re-init branch.
def _make_test_model() -> nn.Module:
    from team_gm.modules.primitives import LayerNorm as ProjectLayerNorm
    from team_gm.modules.primitives import Linear as ProjectLinear

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.proj_default = ProjectLinear(4, 4, init="default")
            self.encoder.proj_zero = ProjectLinear(4, 4, bias=True, init="zero")
            self.encoder.proj_gating = ProjectLinear(4, 4, bias=True, init="gating")
            self.encoder.ln = ProjectLayerNorm(4)
            self.head = nn.Module()
            self.head.linear = nn.Linear(4, 4)
            self.head.rms = nn.RMSNorm(4)

    return M()


# ---------------------------------------------------------------------------
# Prefix-glob matching
# ---------------------------------------------------------------------------


class TestMatches:
    def test_exact(self) -> None:
        assert _matches("a.b.c", "a.b.c")

    def test_prefix_module(self) -> None:
        # Naming a module path covers descendants.
        assert _matches("a.b.c.weight", "a.b.c")
        assert _matches("a.b.c.0.weight", "a.b.c")

    def test_not_a_prefix(self) -> None:
        # ``a.b`` is NOT a substring prefix of ``a.bb`` because of the dot
        # boundary check.
        assert not _matches("a.bb.c", "a.b")

    def test_trailing_dot_normalized(self) -> None:
        assert _matches("a.b.c.weight", "a.b.c.")

    def test_empty_pattern_matches_nothing(self) -> None:
        # Empty pattern is meaningless and must not silently match all.
        assert not _matches("a.b", "")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassify:
    def test_default_when_no_match(self) -> None:
        policy = ParamPolicyConfig(enabled=True, default="reinit")
        assert _classify_one("anything", policy) == "reinit"

    def test_single_match(self) -> None:
        policy = ParamPolicyConfig(enabled=True, freeze=["a.b"])
        assert _classify_one("a.b.weight", policy) == "freeze"

    def test_conflict_raises(self) -> None:
        policy = ParamPolicyConfig(
            enabled=True,
            freeze=["a.b"],
            reinit=["a.b.weight"],  # overlaps with freeze
        )
        with pytest.raises(ParamPolicyConflictError, match="multiple action"):
            _classify_one("a.b.weight", policy)

    def test_classify_params_returns_full_map(self) -> None:
        model = _make_test_model()
        policy = ParamPolicyConfig(
            enabled=True,
            freeze=["encoder.ln"],
            reinit=["encoder.proj_default"],
        )
        out = classify_params(model, policy)
        # Every param accounted for
        assert set(out) == {n for n, _ in model.named_parameters()}
        assert out["encoder.ln.weight"] == "freeze"
        assert out["encoder.proj_default.weight"] == "reinit"
        assert out["head.linear.weight"] == "load_existing"  # default


# ---------------------------------------------------------------------------
# Policy validation (typos)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_typo_raises(self) -> None:
        model = _make_test_model()
        policy = ParamPolicyConfig(enabled=True, freeze=["nonexistent.module"])
        with pytest.raises(ParamPolicyConfigError, match="zero parameters"):
            validate_policy(model, policy)

    def test_conflict_caught(self) -> None:
        model = _make_test_model()
        policy = ParamPolicyConfig(
            enabled=True,
            freeze=["encoder"],
            reinit=["encoder.proj_default"],
        )
        with pytest.raises(ParamPolicyConflictError):
            validate_policy(model, policy)

    def test_disabled_is_noop(self) -> None:
        model = _make_test_model()
        # Garbage policy is fine when enabled=False
        policy = ParamPolicyConfig(enabled=False, freeze=["completely.bogus"])
        validate_policy(model, policy)  # no raise


# ---------------------------------------------------------------------------
# Re-init in-place
# ---------------------------------------------------------------------------


class TestReinit:
    def test_project_linear_zero_init(self) -> None:
        model = _make_test_model()
        # Manually corrupt the zero-init layer's weight
        with torch.no_grad():
            model.encoder.proj_zero.weight.fill_(7.0)
            model.encoder.proj_zero.bias.fill_(3.0)
        policy = ParamPolicyConfig(
            enabled=True,
            reinit=["encoder.proj_zero"],
        )
        apply_param_policy(model, ckpt_state_dict=None, policy=policy)
        # init="zero" => weight zeros, bias zeros
        assert torch.equal(
            model.encoder.proj_zero.weight, torch.zeros_like(model.encoder.proj_zero.weight),
        )
        assert torch.equal(
            model.encoder.proj_zero.bias, torch.zeros_like(model.encoder.proj_zero.bias),
        )

    def test_project_linear_gating_init(self) -> None:
        model = _make_test_model()
        with torch.no_grad():
            model.encoder.proj_gating.weight.fill_(7.0)
            model.encoder.proj_gating.bias.fill_(3.0)
        policy = ParamPolicyConfig(
            enabled=True,
            reinit=["encoder.proj_gating"],
        )
        apply_param_policy(model, ckpt_state_dict=None, policy=policy)
        # init="gating" => weight zeros, bias ones
        assert torch.equal(
            model.encoder.proj_gating.weight, torch.zeros_like(model.encoder.proj_gating.weight),
        )
        assert torch.equal(
            model.encoder.proj_gating.bias, torch.ones_like(model.encoder.proj_gating.bias),
        )

    def test_layernorm_resets_to_ones_zeros(self) -> None:
        model = _make_test_model()
        with torch.no_grad():
            model.encoder.ln.weight.fill_(7.0)
            model.encoder.ln.bias.fill_(3.0)
        policy = ParamPolicyConfig(
            enabled=True,
            reinit=["encoder.ln"],
        )
        apply_param_policy(model, ckpt_state_dict=None, policy=policy)
        assert torch.equal(model.encoder.ln.weight, torch.ones_like(model.encoder.ln.weight))
        assert torch.equal(model.encoder.ln.bias, torch.zeros_like(model.encoder.ln.bias))

    def test_vanilla_linear_resets(self) -> None:
        model = _make_test_model()
        with torch.no_grad():
            model.head.linear.weight.fill_(7.0)
            model.head.linear.bias.fill_(3.0)
        policy = ParamPolicyConfig(
            enabled=True,
            reinit=["head.linear"],
        )
        apply_param_policy(model, ckpt_state_dict=None, policy=policy)
        # Vanilla nn.Linear.reset_parameters => kaiming_uniform, not all 7s.
        assert not torch.allclose(
            model.head.linear.weight, torch.full_like(model.head.linear.weight, 7.0),
        )


# ---------------------------------------------------------------------------
# End-to-end apply_param_policy
# ---------------------------------------------------------------------------


class TestApply:
    def test_load_existing_then_freeze(self) -> None:
        model = _make_test_model()
        # Construct a checkpoint where everything has a known sentinel value
        ckpt = {n: torch.full_like(p, 42.0) for n, p in model.named_parameters()}
        policy = ParamPolicyConfig(
            enabled=True,
            freeze=["encoder.ln"],
            reinit=["encoder.proj_default"],
        )
        summary = apply_param_policy(model, ckpt, policy)

        # encoder.ln: frozen (with ckpt values), requires_grad=False
        assert not model.encoder.ln.weight.requires_grad
        assert torch.allclose(
            model.encoder.ln.weight, torch.full_like(model.encoder.ln.weight, 42.0),
        )
        # encoder.proj_default: reinit'd (NOT 42), requires_grad=True
        assert model.encoder.proj_default.weight.requires_grad
        assert not torch.allclose(
            model.encoder.proj_default.weight,
            torch.full_like(model.encoder.proj_default.weight, 42.0),
        )
        # head.linear: default = load_existing (==42), requires_grad=True
        assert model.head.linear.weight.requires_grad
        assert torch.allclose(
            model.head.linear.weight, torch.full_like(model.head.linear.weight, 42.0),
        )

        assert any(n.startswith("encoder.ln") for n in summary["frozen"])
        assert "encoder.proj_default.weight" in summary["reinit"]
        assert "head.linear.weight" in summary["loaded"]

    def test_missing_ckpt_entry_falls_back_to_reinit(self) -> None:
        model = _make_test_model()
        ckpt = {}  # totally empty
        policy = ParamPolicyConfig(enabled=True)  # default load_existing
        summary = apply_param_policy(model, ckpt, policy)
        # All params end up reinit'd
        assert summary["loaded"] == []
        assert len(summary["reinit"]) == len(list(model.named_parameters()))

    def test_shape_mismatch_falls_back_to_reinit(self) -> None:
        model = _make_test_model()
        # checkpoint with wrong shape for one entry
        ckpt = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        ckpt["encoder.proj_default.weight"] = torch.zeros(5, 5)  # wrong shape
        policy = ParamPolicyConfig(enabled=True)
        summary = apply_param_policy(model, ckpt, policy)
        # The mismatched param ends up in reinit, not loaded
        assert "encoder.proj_default.weight" in summary["reinit"]
        assert "encoder.proj_default.weight" not in summary["loaded"]

    def test_disabled_raises_when_apply_called(self) -> None:
        model = _make_test_model()
        policy = ParamPolicyConfig(enabled=False)
        with pytest.raises(ParamPolicyConfigError):
            apply_param_policy(model, None, policy)


# ---------------------------------------------------------------------------
# freeze_loaded default: freeze what is in the checkpoint, train what is new
# ---------------------------------------------------------------------------


class TestFreezeLoaded:
    def test_present_frozen_missing_trained(self) -> None:
        model = _make_test_model()
        # Checkpoint contains every param EXCEPT the head — emulating a new
        # head attached to a pre-trained trunk.
        ckpt = {
            n: torch.full_like(p, 42.0)
            for n, p in model.named_parameters()
            if not n.startswith("head.")
        }
        policy = ParamPolicyConfig(enabled=True, default="freeze_loaded")
        summary = apply_param_policy(model, ckpt, policy)

        # Present trunk params: loaded (==42) AND frozen.
        assert not model.encoder.ln.weight.requires_grad
        assert torch.allclose(
            model.encoder.ln.weight, torch.full_like(model.encoder.ln.weight, 42.0),
        )
        # Missing head params: reinit'd (!=42) AND trainable.
        assert model.head.linear.weight.requires_grad
        assert not torch.allclose(
            model.head.linear.weight, torch.full_like(model.head.linear.weight, 42.0),
        )

        n_trunk = sum(
            1 for n, _ in model.named_parameters() if not n.startswith("head.")
        )
        n_head = sum(1 for n, _ in model.named_parameters() if n.startswith("head."))
        assert len(summary["frozen"]) == n_trunk
        # Every head param is re-initialized (and none is frozen).
        assert all(n.startswith("head.") or n in summary["frozen"] for n in summary["reinit"])
        assert not any(n.startswith("head.") for n in summary["frozen"])

    def test_missing_freeze_loaded_is_trainable_unlike_plain_freeze(self) -> None:
        # A plain ``freeze`` miss reinits AND freezes; ``freeze_loaded`` miss
        # reinits and stays trainable. Guard the distinction.
        model = _make_test_model()
        policy = ParamPolicyConfig(enabled=True, default="freeze_loaded")
        apply_param_policy(model, {}, policy)  # empty ckpt → everything missing
        assert all(p.requires_grad for p in model.parameters())

    def test_explicit_pattern_overrides_freeze_loaded_default(self) -> None:
        model = _make_test_model()
        ckpt = {n: torch.full_like(p, 42.0) for n, p in model.named_parameters()}
        # Even though head is present in ckpt (freeze_loaded would freeze it),
        # an explicit reinit pattern wins → reinit'd and trainable.
        policy = ParamPolicyConfig(
            enabled=True, default="freeze_loaded", reinit=["head.linear"],
        )
        apply_param_policy(model, ckpt, policy)
        assert model.head.linear.weight.requires_grad
        assert not torch.allclose(
            model.head.linear.weight, torch.full_like(model.head.linear.weight, 42.0),
        )
        # A non-overridden present param is still frozen.
        assert not model.encoder.ln.weight.requires_grad


# ---------------------------------------------------------------------------
# trainable_parameters
# ---------------------------------------------------------------------------


class TestTrainableParameters:
    def test_excludes_frozen(self) -> None:
        model = _make_test_model()
        policy = ParamPolicyConfig(enabled=True, freeze=["encoder"])
        apply_param_policy(model, None, policy)  # missing ckpt -> all reinit'd, then frozen for encoder
        trainables = trainable_parameters(model)
        all_params = list(model.parameters())
        n_frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
        assert len(trainables) == len(all_params) - n_frozen
        assert n_frozen > 0  # sanity


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------


def test_format_summary_truncates() -> None:
    summary = {
        "loaded": [f"p{i}" for i in range(10)],
        "reinit": [],
        "frozen": ["one"],
    }
    text = format_summary(summary, max_per_group=3)
    assert "loaded: 10 params" in text
    assert "(7 more)" in text
    assert "reinit: 0 params" in text
    assert "frozen: 1 params" in text
