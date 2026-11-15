"""F19: NaN/Inf losses or gradients raise instead of applying an update; finite updates unchanged.

The equivalence test compares against the pre-F19 model sources. It reads F19_BASELINE_DIR
(default audit_tmp/remediation/baseline) and is skipped if that copy is absent.
"""
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path(os.environ.get("F19_BASELINE_DIR", ROOT / "audit_tmp" / "remediation" / "baseline"))
N, A, D, F = 33, 5, 10, 7


def batch(B=8, seed=0, reward_value=None):
    g = torch.Generator().manual_seed(seed)
    ei = torch.stack([torch.arange(N - 1), torch.arange(1, N)])
    ei = torch.cat([ei, ei.flip(0)], 1)
    b = dict(x=torch.randn(B, N, F, generator=g), next_x=torch.randn(B, N, F, generator=g), edge_index=ei,
             agent_obs=torch.randn(B, A, D, generator=g), next_agent_obs=torch.randn(B, A, D, generator=g),
             actions=torch.rand(B, A, 1, generator=g) * 2 - 1, rewards=torch.randn(B, A, generator=g),
             dones=torch.zeros(B, 1))
    if reward_value is not None:
        b["rewards"][0, 0] = reward_value
    return b


def make(kind, module=None):
    torch.manual_seed(0)
    if kind == "mlp":
        mod = module or importlib.import_module("experiments.no_gnn_maddpg.maddpg_mlp")
        return mod.MLPMADDPG(agent_obs_dim=D, num_agents=A, device="cpu", actor_hidden_dims=(32,),
                             critic_hidden_dims=(32,))
    mod = module or importlib.import_module("models.maddpg")
    return mod.GraphAwareMADDPG(node_feature_dim=F, agent_obs_dim=D, dess_buses=list(range(1, A + 1)),
                                num_agents=A, device="cpu", gnn_hidden_dim=16, gnn_embedding_dim=16,
                                gnn_num_layers=2, actor_hidden_dims=(32,), critic_hidden_dims=(32,))


def params(agent):
    return [p.detach().clone() for m in (agent.actor, agent.critics, agent.target_actor, agent.target_critics)
            for p in m.parameters()]


KINDS = ["mlp"]
try:
    import torch_geometric  # noqa: F401
    KINDS.append("gcn")
except ImportError:
    pass


class NonFiniteSafeguards(unittest.TestCase):
    def test_nonfinite_loss_raises_without_update(self):
        for kind in KINDS:
            for bad in (float("nan"), float("inf")):
                with self.subTest(kind=kind, bad=bad):
                    agent = make(kind)
                    before = params(agent)
                    with self.assertRaisesRegex(FloatingPointError, "critic 0 loss"):
                        agent.update(batch(reward_value=bad))
                    for p, q in zip(before, params(agent)):
                        self.assertTrue(torch.equal(p, q))

    def test_nonfinite_gradient_refuses_step(self):
        for kind in KINDS:
            with self.subTest(kind=kind, part="critic"):
                agent = make(kind)
                before = [p.detach().clone() for p in agent.critics[0].parameters()]
                handle = next(agent.critics[0].parameters()).register_hook(lambda g: g * float("nan"))
                with self.assertRaisesRegex(FloatingPointError, "critic 0 gradient norm"):
                    agent.update(batch())
                handle.remove()
                for p, q in zip(before, agent.critics[0].parameters()):
                    self.assertTrue(torch.equal(p, q.detach()))
                self.assertTrue(all(p.grad is None for p in agent.critics[0].parameters()))
            with self.subTest(kind=kind, part="actor"):
                agent = make(kind)
                before = [p.detach().clone() for p in agent.actor.parameters()]
                handle = next(agent.actor.parameters()).register_hook(lambda g: g * float("nan"))
                with self.assertRaisesRegex(FloatingPointError, "actor gradient norm"):
                    agent.update(batch())
                handle.remove()
                for p, q in zip(before, agent.actor.parameters()):
                    self.assertTrue(torch.equal(p, q.detach()))

    @unittest.skipUnless((BASE / "models" / "maddpg.py").is_file(), f"pre-F19 baseline not found: {BASE}")
    def test_finite_updates_identical_to_pre_f19(self):
        sources = {"gcn": BASE / "models" / "maddpg.py", "mlp": BASE / "experiments" / "no_gnn_maddpg" / "maddpg_mlp.py"}
        for kind in KINDS:
            with self.subTest(kind=kind):
                spec = importlib.util.spec_from_file_location(f"pre_f19_{kind}", sources[kind])
                old_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(old_mod)
                new, old = make(kind), make(kind, old_mod)
                for p, q in zip(params(new), params(old)):
                    self.assertTrue(torch.equal(p, q))
                for step in range(3):
                    log_new, log_old = new.update(batch(seed=step)), old.update(batch(seed=step))
                    self.assertEqual(log_new["critic_loss"], log_old["critic_loss"])
                    self.assertEqual(log_new["actor_loss"], log_old["actor_loss"])
                for p, q in zip(params(new), params(old)):
                    self.assertTrue(torch.equal(p, q))


if __name__ == "__main__":
    unittest.main()
