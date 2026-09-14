"""Model-shape and parameter-budget tests.

The prior attempt put 14.4M image-branch parameters against ~5,300 training
samples and overfit badly. The budget is asserted here so growing the encoder is a
deliberate act, not an accident.
"""
import pytest
import torch

from models.fusion import VARIANTS, MultimodalNowcaster
from models.spatial import SpatialEncoder
from models.temporal import TemporalEncoder


def make_batch(b=4, L=24, F=11, C=4, H=64, W=64, nwp=3):
    return {"seq": torch.randn(b, L, F), "grid": torch.randn(b, C, H, W),
            "gate": torch.randn(b, 4), "future_cs": torch.rand(b, 3),
            "nwp": torch.randn(b, nwp)}


def test_spatial_encoder_stays_under_one_million_params(cfg):
    enc = SpatialEncoder(4, tuple(cfg["model"]["spatial"]["channels"]),
                         cfg["model"]["spatial"]["dropout"],
                         out_dim=cfg["model"]["fusion"]["dim"])
    n = enc.n_params()
    assert n < 1_000_000, f"spatial encoder has {n:,} params, budget is <1M"


def test_total_params_are_sane_against_sample_count(cfg):
    m = MultimodalNowcaster(11, 4, 3, 5, n_nwp=3, variant="gated", cfg=cfg)
    total = m.n_params()["total"]
    assert total < 2_000_000, f"model has {total:,} params; the prior overfit at 14.4M"


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_output_shape(cfg, variant):
    m = MultimodalNowcaster(11, 4, 3, 5, n_nwp=3, variant=variant, cfg=cfg)
    out = m(make_batch())
    assert out.shape == (4, 3, 5)


def test_temporal_only_ignores_the_grid(cfg):
    """A temporal-only ablation must not be secretly reading the rain field."""
    torch.manual_seed(0)
    m = MultimodalNowcaster(11, 4, 3, 5, n_nwp=3, variant="temporal", cfg=cfg).eval()
    b = make_batch()
    a = m(b)
    b["grid"] = torch.randn_like(b["grid"]) * 50
    assert torch.allclose(a, m(b), atol=1e-6), "temporal variant used the grid"


def test_spatial_only_ignores_the_sequence(cfg):
    torch.manual_seed(0)
    m = MultimodalNowcaster(11, 4, 3, 5, n_nwp=3, variant="spatial", cfg=cfg).eval()
    b = make_batch()
    a = m(b)
    b["seq"] = torch.randn_like(b["seq"]) * 50
    assert torch.allclose(a, m(b), atol=1e-6), "spatial variant used the sequence"


def test_gate_alpha_is_recorded_and_bounded(cfg):
    m = MultimodalNowcaster(11, 4, 3, 5, n_nwp=3, variant="gated", cfg=cfg)
    m(make_batch(b=8))
    a = m.last_alpha
    assert a is not None and a.shape == (8, 1)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_non_gated_variants_expose_no_alpha(cfg):
    for v in ("temporal", "spatial", "concat"):
        m = MultimodalNowcaster(11, 4, 3, 5, n_nwp=3, variant=v, cfg=cfg)
        m(make_batch())
        assert m.last_alpha is None, f"{v} should not produce a gate alpha"


def test_model_works_without_nwp(cfg):
    """Track B carries no NWP branch; the head must still build."""
    m = MultimodalNowcaster(11, 4, 3, 5, n_nwp=0, variant="gated", cfg=cfg)
    b = make_batch(nwp=0)
    b["nwp"] = torch.zeros(4, 0)
    assert m(b).shape == (4, 3, 5)


def test_bad_variant_is_rejected(cfg):
    with pytest.raises(ValueError):
        MultimodalNowcaster(11, 4, 3, 5, variant="not_a_variant", cfg=cfg)


def test_temporal_encoder_is_bidirectional():
    enc = TemporalEncoder(11, hidden=32)
    assert enc.out_dim == 64
    summary, steps = enc(torch.randn(4, 24, 11))
    assert summary.shape == (4, 64) and steps.shape == (4, 24, 64)
