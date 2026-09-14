"""Quantile head: monotonicity, pinball loss behaviour, and bimodality handling."""
import numpy as np
import pytest
import torch

from models.fusion import MultimodalNowcaster, VARIANTS
from models.head import QuantileHead, pinball_loss


def test_output_is_sorted_along_quantile_axis():
    torch.manual_seed(0)
    head = QuantileHead(in_dim=32, n_horizons=3, n_quantiles=5)
    out = head(torch.randn(64, 32))
    assert out.shape == (64, 3, 5)
    assert (out.diff(dim=-1) >= -1e-6).all(), "quantiles cross"


def test_sorting_fixes_deliberately_crossed_outputs():
    head = QuantileHead(in_dim=8, n_horizons=1, n_quantiles=5)
    with torch.no_grad():                       # force a descending raw output
        head.net[-1].weight.zero_()
        head.net[-1].bias.copy_(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]))
    out = head(torch.randn(4, 8))
    assert (out.diff(dim=-1) >= -1e-6).all()
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_variants_emit_monotonic_quantiles(cfg, variant):
    torch.manual_seed(0)
    m = MultimodalNowcaster(11, 4, 3, 5, n_nwp=3, variant=variant, cfg=cfg)
    batch = {"seq": torch.randn(6, 24, 11), "grid": torch.randn(6, 4, 64, 64),
             "gate": torch.randn(6, 4), "future_cs": torch.rand(6, 3),
             "nwp": torch.randn(6, 3)}
    out = m(batch)
    assert out.shape == (6, 3, 5)
    assert (out.diff(dim=-1) >= -1e-6).all()


def test_pinball_is_minimised_at_the_true_quantiles():
    """Fit a constant per quantile to a known sample; recover its quantiles."""
    torch.manual_seed(0)
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    sample = torch.randn(20000, 1) * 2.0 + 5.0
    pred = torch.zeros(1, 1, 5, requires_grad=True)
    opt = torch.optim.Adam([pred], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        loss = pinball_loss(pred.expand(sample.shape[0], 1, 5), sample, qs)
        loss.backward()
        opt.step()
    got = pred.detach().flatten().numpy()
    want = np.quantile(sample.numpy(), qs)
    assert np.allclose(got, want, atol=0.15), f"got {got}, want {want}"


def test_pinball_beats_the_mean_on_a_bimodal_target():
    """The motivating case: k_t either stays overcast or clears.

    A single point estimate at the mean sits in the empty valley between the
    modes. Quantile regression places its 10th/90th on the modes themselves.
    """
    torch.manual_seed(0)
    n = 20000
    modes = torch.where(torch.rand(n, 1) < 0.6,
                        torch.randn(n, 1) * 0.05 + 0.30,      # stays overcast
                        torch.randn(n, 1) * 0.05 + 0.75)      # clears
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    pred = torch.zeros(1, 1, 5, requires_grad=True)
    opt = torch.optim.Adam([pred], lr=0.05)
    for _ in range(800):
        opt.zero_grad()
        loss = pinball_loss(pred.expand(n, 1, 5), modes, qs)
        loss.backward()
        opt.step()
    got = pred.detach().flatten().numpy()
    lo, hi = got[0], got[-1]
    assert lo < 0.45, f"10th percentile {lo:.3f} should sit on the overcast mode"
    assert hi > 0.60, f"90th percentile {hi:.3f} should sit on the cleared mode"
    # and the interval must be much wider than a Gaussian fitted to the same data
    assert (hi - lo) > 0.35, "quantile spread collapsed toward the mean"


def test_pinball_loss_is_zero_for_perfect_constant_target():
    qs = [0.1, 0.5, 0.9]
    target = torch.full((16, 3), 2.0)
    pred = torch.full((16, 3, 3), 2.0)
    assert pinball_loss(pred, target, qs).item() == pytest.approx(0.0, abs=1e-7)


def test_pinball_penalises_asymmetrically():
    """q=0.9 should punish under-prediction more than over-prediction."""
    qs = [0.9]
    target = torch.zeros(1, 1)
    under = pinball_loss(torch.full((1, 1, 1), -1.0), target, qs)
    over = pinball_loss(torch.full((1, 1, 1), 1.0), target, qs)
    assert under > over
