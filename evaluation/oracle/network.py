"""LinDistFlow (lossless, linearised branch-flow) model of the environment's radial network.

Built from exactly what the AC simulator uses (utility.power_flow.TensorPowerFlow of the topology-specific env):
in-service lines (FROM/TO 1-based, R/X in ohm), z_base, s_base, slack bus (|V| = 1.0 pu), and the env's line-current
limits (same in-service line order as the simulated currents). Shunt B and taps are ignored (LinDistFlow).

With p_k, q_k the net active/reactive demand of bus k in pu (load - PV - battery; reactive = load kvar only):
    branch flow    F_e = sum_{k downstream of e} (p_k, q_k)                  = Path @ p
    squared volt.  w_b = 1 - 2 sum_k (Rpath[b,k] p_k + Xpath[b,k] q_k),     Rpath = Path^T diag(r) Path
Path[e, k] = 1 if line e lies on the slack-to-k path. Losses are neglected, so upstream flows and voltage drops are
underestimated; the solver therefore applies explicit voltage and branch-loading margins.
"""
from collections import deque

import numpy as np


class LinDistFlowNetwork:
    def __init__(self, env):
        pf = env.power_flow
        lines = pf.in_service_line_info
        self.num_buses = int(env.num_buses)
        self.slack = int(pf.slack_bus)
        self.s_base_kva = float(pf.s_base_kva)
        self.v_min, self.v_max = float(env.v_min), float(env.v_max)
        ends = [(int(r['FROM']) - 1, int(r['TO']) - 1) for _, r in lines.iterrows()]
        r_pu = lines['R'].to_numpy(float) / pf.z_base_ohm
        x_pu = lines['X'].to_numpy(float) / pf.z_base_ohm
        if len(ends) != self.num_buses - 1:
            raise ValueError(f'{len(ends)} in-service lines for {self.num_buses} buses: network is not radial')
        adjacency = {b: [] for b in range(self.num_buses)}
        for e, (i, j) in enumerate(ends):
            adjacency[i].append((j, e))
            adjacency[j].append((i, e))
        parent_line = {self.slack: None}
        queue = deque([self.slack])
        while queue:
            bus = queue.popleft()
            for nxt, e in adjacency[bus]:
                if nxt not in parent_line:
                    parent_line[nxt] = e
                    queue.append(nxt)
        if len(parent_line) != self.num_buses:
            raise ValueError('In-service network is not connected')
        # Orientation: each line feeds its downstream ("child") bus; from_bus = upstream end.
        self.child = np.empty(len(ends), dtype=int)
        self.upstream = np.empty(len(ends), dtype=int)
        for bus, e in parent_line.items():
            if e is not None:
                i, j = ends[e]
                self.child[e], self.upstream[e] = bus, (i if j == bus else j)
        self.path = np.zeros((len(ends), self.num_buses))
        for k in range(self.num_buses):
            bus = k
            while parent_line[bus] is not None:
                e = parent_line[bus]
                self.path[e, k] = 1.0
                bus = self.upstream[e]
        self.r_path = self.path.T @ (r_pu[:, None] * self.path)
        self.x_path = self.path.T @ (x_pu[:, None] * self.path)
        self.r_pu, self.x_pu = r_pu, x_pu
        self.num_lines = len(ends)
        # Line-current limits of the env (pu of the env's current base), same order as the simulated line currents.
        self.current_limit_pu = np.asarray(env.grid.get_line_current_limits_pu(), dtype=float)
        if self.current_limit_pu.shape != (self.num_lines,):
            raise ValueError('Line-current limits do not match the in-service lines')
        self.non_slack = np.array([b for b in range(self.num_buses) if b != self.slack])

    def squared_voltage(self, p_pu, q_pu):
        """w = 1 - 2 (Rpath p + Xpath q) for net demand vectors (pu); returns |V| estimates (sqrt of w)."""
        w = 1.0 - 2.0 * (self.r_path @ p_pu + self.x_path @ q_pu)
        return np.sqrt(np.maximum(w, 0.0))

    def branch_flow(self, p_pu, q_pu):
        return self.path @ p_pu, self.path @ q_pu
