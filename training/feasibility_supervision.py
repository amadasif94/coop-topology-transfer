"""Pre-tanh feasible-action supervision for B3 actors, and its training diagnostics.

Every original actor computes its action as a = tanh(z), with z = actors[i].net(x) for the per-agent network of
MLP A0/A1/A2 (MLPDESSActor) and GCN/GAT/TAGConv (DESSActor). A forward hook on each per-agent `net` receives that
exact z, the tensor tanh consumes, inside the controller's ORIGINAL update(). No pinned file is modified, and no
atanh of an action is ever differentiated.

Feasible normalised limits [l, h] (pre-action, replay-stored, -1 <= l <= 0 <= h <= 1) map to pre-tanh limits
    z_low = atanh(l)  (= -inf when l = -1),   z_high = atanh(h)  (= +inf when h = +1)
so a side touching +-1 is unbounded, and a zero-width interval l = h = 0 gives z_low = z_high = 0.
Feasibility loss (mean over sampled batch b and agents i):
    L_feas = mean_{b,i} [ relu(z - z_high)^2 + relu(z_low - z)^2 ]
Actor objective: L_actor = L_actor^original + lambda_f * L_feas. A gradient hook on each z adds
lambda_f * dL_feas/dz = lambda_f * 2 * (relu(z - z_high) - relu(z_low - z)) / (B * N) to the incoming policy
gradient, which is exactly the gradient of the sum. It then flows through the original backward, the original gradient
clipping and the original optimizer step. Unlike the historical post-tanh loss, this gradient is not multiplied by
1 - a^2, so it does not vanish when the actor saturates. Critic losses, targets, the target actor and replay are
untouched.

Diagnostics (identical definitions for all six controllers):
  every update      z statistics (mean |z|, std, |z| quantiles, max), post-tanh saturation (|a| > 0.99 / 0.999),
                    deterministic actor infeasible fraction (current actor on sampled replay states vs stored bounds),
                    pre-tanh feasibility loss, post-tanh action-space violation (comparison with the historical
                    loss), actor gradient norm BEFORE clipping (parameter hooks) and AFTER clipping (.grad)
  diagnostic update Q-term and lambda-weighted feasibility-term actor-parameter gradient norms (snapshot of the actor
                    before the update; same parameters as the update), and per-element z-gradients of both terms;
                    the same two norms per agent (own actor parameters) and their ratio rho (gradient balance)
"""
from copy import deepcopy
import math

import torch

FORMULATION = 'pre_tanh_squared_hinge'
RHO_EPS = 1e-12          # gradient-balance ratio: ||grad L_RL|| / (||grad lambda_f L_feas|| + RHO_EPS)
HISTORICAL_FORMULATION = 'post_tanh_action_space_squared_hinge'


def pretanh_bounds(lower, upper):
    """(z_low, z_high) for normalised limits; computed in float64, +-inf where the limit is +-1."""
    lo = torch.atanh(torch.clamp(lower.double(), -1.0, 0.0))
    hi = torch.atanh(torch.clamp(upper.double(), 0.0, 1.0))
    return lo.to(lower.dtype), hi.to(upper.dtype)


def pretanh_violation(z, z_low, z_high):
    """relu(z - z_high) - relu(z_low - z): positive above the discharge limit, negative below the charge limit."""
    return torch.relu(z - z_high) - torch.relu(z_low - z)


def pretanh_feasibility_loss(z, z_low, z_high):
    return (pretanh_violation(z, z_low, z_high) ** 2).mean()


def actor_networks(actor, num_agents):
    """The per-agent networks whose output is the pre-tanh z (separate actors are required)."""
    nets = [a.net for a in getattr(actor, 'actors', [])]
    if len(nets) != num_agents:
        raise ValueError('B3 pre-tanh supervision needs one separate per-agent actor network per agent (share_actor=False)')
    return nets


def _norm(grads):
    return math.sqrt(float(sum((g.double() ** 2).sum() for g in grads if g is not None)))


class FeasibilitySupervision:
    """Context for one original update(): pre-tanh supervision hooks plus diagnostics."""

    def __init__(self, model, batch, coefficient, detailed=False, attach_loss=True):
        """attach_loss=False (soc_feasible: no SoC-feasibility supervision) registers NO gradient modification: the
        z gradient hook only reads dL_actor/dz, the coefficient must be 0, and the update is exactly the original."""
        if not attach_loss and float(coefficient) != 0.0:
            raise ValueError('diagnostics-only supervision requires lambda_f = 0')
        self.model, self.coefficient, self.detailed = model, float(coefficient), detailed
        self.attach_loss = bool(attach_loss)
        lower = model._to_device(batch['feasible_action_lower'])
        upper = model._to_device(batch['feasible_action_upper'])
        self.B, self.N = lower.shape
        if self.N != model.num_agents:
            raise ValueError(f'Bounds for {self.N} agents, model has {model.num_agents}')
        self.lower, self.upper = lower, upper
        self.z_low, self.z_high = pretanh_bounds(lower, upper)
        self.batch = batch
        self.z, self.q_grad, self.fired = {}, {}, 0
        self.pre_clip_sq = 0.0
        self._handles = []
        self.snapshot = None

    # ----------------------------------------------------------------------------------- hooks
    def __enter__(self):
        if self.detailed:   # the actor exactly as the update will use it (actor parameters change only at its step)
            self.snapshot = deepcopy(self.model.actor)
            self.snap_z = self._forward_z(self.snapshot)
        for i, net in enumerate(actor_networks(self.model.actor, self.N)):
            self._handles.append(net.register_forward_hook(self._make_hook(i)))
        for p in self.model.actor.parameters():
            if p.requires_grad:
                self._handles.append(p.register_hook(self._param_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        if exc[0] is None and self.fired != self.N:
            raise RuntimeError(f'B3 pre-tanh hook fired {self.fired} times, expected {self.N}')
        return False

    def _make_hook(self, i):
        def hook(module, args, z):
            if not torch.is_grad_enabled() or not z.requires_grad:
                return None
            self.fired += 1
            col = z.reshape(self.B, -1)
            if col.shape != (self.B, 1):
                raise ValueError(f'Pre-tanh output of agent {i} has shape {tuple(z.shape)}, expected [{self.B}, 1]')
            v = pretanh_violation(col.detach(), self.z_low[:, i:i + 1], self.z_high[:, i:i + 1])
            self.z[i] = col.detach()
            extra = (self.coefficient * 2.0 * v / (self.B * self.N)).reshape(z.shape)

            def grad_hook(g):
                self.q_grad[i] = g.detach().reshape(self.B, -1)       # dL_actor^original / dz_i (already through tanh)
                return g + extra

            def read_only_hook(g):
                self.q_grad[i] = g.detach().reshape(self.B, -1)
                return None                                            # gradient left exactly as the original update
            z.register_hook(grad_hook if self.attach_loss else read_only_hook)
            return None
        return hook

    def _param_hook(self, g):
        self.pre_clip_sq += float((g.double() ** 2).sum())
        return None

    def _forward_z(self, actor):
        captured = {}
        handles = [net.register_forward_hook(lambda m, a, out, i=i: captured.__setitem__(i, out))
                   for i, net in enumerate(actor_networks(actor, self.N))]
        try:
            self.model._actor_forward(actor, self.batch)
        finally:
            for h in handles:
                h.remove()
        return torch.cat([captured[i].reshape(self.B, -1) for i in range(self.N)], 1)

    # ------------------------------------------------------------------------------- summaries
    def results(self):
        z = torch.cat([self.z[i] for i in range(self.N)], 1)
        a = torch.tanh(z)
        v = pretanh_violation(z, self.z_low, self.z_high)
        outside = v != 0
        action_violation = a - torch.maximum(torch.minimum(a, self.upper), self.lower)
        absz = z.abs().flatten().double()
        q = torch.quantile(absz, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64, device=absz.device))
        grads = [p.grad for p in self.model.actor.parameters() if p.grad is not None]
        out = dict(feasibility_formulation=FORMULATION if self.attach_loss else 'none',
                   lambda_feasibility=self.coefficient,
                   feasibility_loss=float((v ** 2).mean()), feasibility_loss_weighted=self.coefficient * float((v ** 2).mean()),
                   action_space_violation_sq=float((action_violation ** 2).mean()),
                   actor_infeasible_fraction=float(outside.float().mean()),
                   preact_abs_mean=float(absz.mean()), preact_std=float(z.std()), preact_abs_p50=float(q[0]),
                   preact_abs_p90=float(q[1]), preact_abs_p99=float(q[2]), preact_abs_max=float(absz.max()),
                   saturation_fraction_099=float((a.abs() > 0.99).float().mean()),
                   saturation_fraction_0999=float((a.abs() > 0.999).float().mean()),
                   actor_grad_norm_pre_clip=math.sqrt(self.pre_clip_sq), actor_grad_norm_post_clip=_norm(grads))
        capture = dict(z=z, actions=a, violation=v, outside=outside)
        if len(self.q_grad) == self.N:
            g = torch.cat([self.q_grad[i] for i in range(self.N)], 1) * (self.B * self.N)   # per-sample dL_Q/dz
            capture['q_z_grad'] = g
            if outside.any():
                out.update(q_z_grad_median_abs_infeasible=float(g[outside].abs().median()),
                           feas_z_grad_median_abs_infeasible=float((2.0 * v[outside]).abs().median()))
        if self.detailed:
            if not torch.allclose(self.snap_z.detach(), z, atol=1e-5, rtol=1e-5):
                ref = self.snap_z.detach()
                actual = z.detach()
                finite = torch.isfinite(ref) & torch.isfinite(actual)
                differences = (ref - actual).abs()
                finite_diff = differences[finite]
                mismatch = ~torch.isclose(
                    ref, actual, atol=1e-5, rtol=1e-5
                )
                raise RuntimeError(
                    "Diagnostic actor snapshot differs from the update actor; "
                    f"max_finite_difference={finite_diff.max().item() if finite_diff.numel() else float('nan')}; "
                    f"mismatch_count={int(mismatch.sum().item())}; "
                    f"snapshot_nonfinite={int((~torch.isfinite(ref)).sum().item())}; "
                    f"update_nonfinite={int((~torch.isfinite(actual)).sum().item())}"
                )
            params = [p for p in self.snapshot.parameters() if p.requires_grad]
            loss = pretanh_feasibility_loss(self.snap_z, self.z_low, self.z_high)
            feas = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            out['feasibility_param_grad_norm_unweighted'] = _norm(feas)
            out['feasibility_param_grad_norm'] = self.coefficient * out['feasibility_param_grad_norm_unweighted']
            if 'q_z_grad' in capture:
                qg = torch.autograd.grad(self.snap_z, params, grad_outputs=capture['q_z_grad'] / (self.B * self.N),
                                         allow_unused=True)
                out['q_param_grad_norm'] = _norm(qg)
                # Per-agent gradient balance rho_i = ||grad_i L_RL|| / (||grad_i lambda_f L_feas|| + eps) over agent i's own
                # actor parameters (actors[i]; a shared graph encoder is not attributed to any agent).
                owner = {id(p): i for i, a in enumerate(self.snapshot.actors) for p in a.parameters()}
                q_i, f_i = [0.0] * self.N, [0.0] * self.N
                for p, g_q, g_f in zip(params, qg, feas):
                    if id(p) in owner:
                        i = owner[id(p)]
                        q_i[i] += 0.0 if g_q is None else float((g_q.double() ** 2).sum())
                        f_i[i] += 0.0 if g_f is None else float((g_f.double() ** 2).sum())
                out['q_param_grad_norm_per_agent'] = [math.sqrt(v) for v in q_i]
                out['feasibility_param_grad_norm_per_agent'] = [self.coefficient * math.sqrt(v) for v in f_i]
                out['gradient_balance_per_agent'] = [q / (f + RHO_EPS) for q, f in
                                                     zip(out['q_param_grad_norm_per_agent'],
                                                         out['feasibility_param_grad_norm_per_agent'])]
                out['gradient_balance'] = out['q_param_grad_norm'] / (out['feasibility_param_grad_norm'] + RHO_EPS)
                if not self.attach_loss:            # no feasibility term: the ratio is undefined, not "infinite"
                    out['gradient_balance_per_agent'] = [None] * self.N
                    out['gradient_balance'] = None
        return out, capture
