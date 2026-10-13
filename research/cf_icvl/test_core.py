"""CPU-only mechanical checks. These are not CrowdNav navigation metrics.

Run from the Gmony repository root:
    python -m research.cf_icvl.test_core
"""
import torch
from torch import nn

from .core import (
    IntentConditionedJointCritic, select_joint_decision,
    cf_regret_loss, behavior_clone_loss, td_joint_loss,
    collision_supervision_loss,
)

def main():
    torch.manual_seed(17)
    b, d, a = 5, 16, 80
    encoder = nn.GRU(d, d, batch_first=True)
    critic = IntentConditionedJointCritic(feature_dim=d, n_actions=a)
    seq = torch.randn(b, 24, d)
    history, _ = encoder(seq)
    prev = torch.tensor([0, 1, 1, 2, 0])
    values = critic(history[:, -1], prev)
    assert values.q.shape == (b, 4, a)
    assert values.collision_logit.shape == (b, 4, a)
    assert torch.isfinite(values.q).all()

    safe = torch.ones(b, a, dtype=torch.bool)
    safe[:, 0] = False
    mode_mask = torch.ones(b, 4, a, dtype=torch.bool)
    modes, actions = select_joint_decision(values, safe, mode_mask)
    assert modes.shape == actions.shape == (b,)
    assert not torch.any(actions == 0)

    # Combined IL + RL + counterfactual + risk gradients must reach encoder.
    cf = cf_regret_loss(values, torch.ones(b, dtype=torch.long),
                        torch.zeros(b, dtype=torch.long),
                        torch.ones(b), mode_mask)
    il = behavior_clone_loss(values, torch.randint(0, a, (b,)))
    next_values = critic(history[:, -1].detach(), modes.detach())
    rl = td_joint_loss(values, modes.detach(), actions.detach(),
                       torch.randn(b), next_values,
                       torch.tensor([0, 0, 1, 0, 1]))
    risk = collision_supervision_loss(values, modes.detach(), actions.detach(),
                                      torch.tensor([0, 0, 1, 0, 0]))
    (cf + il + rl + risk).backward()
    assert encoder.weight_ih_l0.grad is not None
    assert encoder.weight_ih_l0.grad.abs().sum() > 0
    assert critic.q_head.weight.grad is not None
    assert critic.collision_head.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None)

    # Paired counterfactual supervision should increase the learned option gap.
    small = IntentConditionedJointCritic(feature_dim=8, n_actions=4, hidden=20)
    optimizer = torch.optim.Adam(small.parameters(), lr=0.01)
    feature = torch.randn(10, 8)
    old_intent = torch.zeros(10, dtype=torch.long)
    all_valid = torch.ones(10, 4, 4, dtype=torch.bool)
    def gap():
        q = small(feature, old_intent).q.max(-1).values
        return (q[:, 1] - q[:, 0]).mean().item()
    before = gap()
    for _ in range(35):
        optimizer.zero_grad()
        loss = cf_regret_loss(small(feature, old_intent),
                              torch.ones(10, dtype=torch.long), old_intent,
                              torch.ones(10), all_valid)
        loss.backward()
        optimizer.step()
    after = gap()
    assert after > before + 0.2, (before, after)
    print("PASS: joint shapes; action feasibility; IL+TD+CF+risk gradients; paired preference fitting")
    print("Learned preference gap (before/after):", round(before, 3), round(after, 3))

if __name__ == "__main__":
    main()
