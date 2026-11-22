"""Leaky integrate-and-fire network built from connectome weights.

Neuron/synapse constants follow Shiu et al. 2024 (Nature), "A Drosophila
computational brain model reveals sensorimotor processing":
tau_m 20 ms, V_rest = V_reset -52 mV, V_th -45 mV, refractory 2.2 ms,
tau_syn 5 ms, synaptic delay 1.8 ms, 0.275 mV per synapse.
Weights are synapse counts signed by predicted neurotransmitter
(GABA/glutamate inhibitory, acetylcholine excitatory).
"""
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent / "data"


class Brain:
    TAU_M, V_REST, V_TH, T_REF = 20.0, -52.0, -45.0, 2.2
    TAU_SYN, DELAY, W_SYN = 5.0, 1.8, 0.275

    def __init__(self, dt=0.5, seed=0, w_scale=1.0):
        z = np.load(DATA / "circuit.npz", allow_pickle=True)
        self.W = z["W"] * self.W_SYN * w_scale          # pre x post, mV
        self.cell_type = z["cell_type"]
        self.groups = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
        self.n = self.W.shape[0]
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.n_delay = max(1, int(round(self.DELAY / dt)))
        self.silenced = np.zeros(self.n, bool)
        self.reset()

    def reset(self):
        self.v = np.full(self.n, self.V_REST)
        self.g = np.zeros(self.n)
        self.ref = np.zeros(self.n)
        self.buf = [np.zeros(self.n) for _ in range(self.n_delay)]
        self.counts = np.zeros(self.n)

    def silence(self, *group_names):
        for g in group_names:
            self.silenced[self.groups[g]] = True

    def step(self, drive):
        """drive: dict group_name -> Poisson rate (Hz) that forces input neurons to spike."""
        dt = self.dt
        self.g += self.buf.pop(0)
        self.v += dt * (self.g - (self.v - self.V_REST)) / self.TAU_M
        self.g -= dt * self.g / self.TAU_SYN
        self.ref -= dt
        self.v[self.ref > 0] = self.V_REST
        spk = self.v >= self.V_TH
        for name, rate in drive.items():
            ix = self.groups[name]
            spk[ix] |= self.rng.random(len(ix)) < rate * dt * 1e-3
        spk &= ~self.silenced
        spk &= self.ref <= 0
        self.v[spk] = self.V_REST
        self.ref[spk] = self.T_REF
        self.buf.append(self.W[spk].sum(0) if spk.any() else np.zeros(self.n))
        self.counts += spk
        return spk

    def run(self, drive, ms):
        """Run for `ms` milliseconds; return per-neuron firing rate (Hz) in that window."""
        self.counts[:] = 0
        steps = int(ms / self.dt)
        for _ in range(steps):
            self.step(drive)
        return self.counts / (ms * 1e-3)

    def rate(self, rates, group):
        return rates[self.groups[group]].mean()
