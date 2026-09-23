import torch

from jevforge.rlcd import distribution_metrics, rlcd_loss, utility_rewards


def test_utility_rewards_preserve_multiple_valid_actions():
    target = torch.tensor([0.5, 0.5, 0.0])
    assert torch.equal(utility_rewards(target), torch.tensor([1.0, 1.0, 0.0]))


def test_exact_objective_prefers_gold_and_backpropagates():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    target = torch.tensor([0.0, 1.0])
    reference = torch.tensor([0.0, 0.0])
    loss, metrics = rlcd_loss(logits, target, reference, mode="exact",
                              calibration_weight=0.5, kl_weight=0.02)
    loss.backward()
    assert logits.grad[1] < 0
    assert metrics["utility"].item() == 0.5


def test_kl_is_zero_for_reference_distribution():
    logits = torch.tensor([-0.3, 0.7, 0.1])
    target = torch.tensor([0.0, 1.0, 0.0])
    metrics = distribution_metrics(logits, target, logits.clone())
    assert abs(metrics["kl"].item()) < 1e-7


def test_sampled_objective_is_finite():
    logits = torch.tensor([0.2, -0.2, 0.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 0.0])
    generator = torch.Generator().manual_seed(7)
    loss, metrics = rlcd_loss(logits, target, logits.detach(), mode="sampled",
                              group_size=32, generator=generator)
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["sampled_reward"].item() <= 1.0
