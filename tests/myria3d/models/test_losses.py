import torch

from myria3d.models.losses import (
    WeightedKLDivLoss,
    kl_divergence_rows,
    pool_axis_distribution_from_probs,
)


def test_pool_axis_distribution_and_weighted_kl():
    # Two tiles: 3 points then 2 points. Class 2 is ignore_index.
    probs = torch.tensor(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [0.0, 1.0],  # ignored
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    target = torch.tensor([0, 1, 2, 0, 0])
    ptr = torch.tensor([0, 3, 5])

    pi_hat, q_t, n_t = pool_axis_distribution_from_probs(
        probs, target, ptr, ignore_index=2, num_classes=2
    )

    assert n_t.tolist() == [2.0, 2.0]
    torch.testing.assert_close(q_t[0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(q_t[1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(pi_hat[0], torch.tensor([0.55, 0.45]))
    torch.testing.assert_close(pi_hat[1], torch.tensor([1.0, 0.0]))

    expected_kl = kl_divergence_rows(q_t, pi_hat)
    loss = WeightedKLDivLoss()(pi_hat, q_t, weight=n_t)
    torch.testing.assert_close(loss, expected_kl.mean())
