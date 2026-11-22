"""Closed-loop courtship: a connectome-driven male pursues a female in a 2D arena.

Brain <-> world interface (all hand-designed, see README "What is real"):
  * Pheromone: female emits an odor; concentration at each antenna sets the
    Poisson rate of the left/right pheromone ORNs (Or67d/DA1 + Or47b/VA1v).
  * Contact: within touching distance the fru+ ascending neurons fire
    (stand-in for foreleg ppk23 taste neurons, which live in the VNC).
  * Vision: female's bearing and angular size drive left/right LC10a.
  * Arousal: pC1 is driven by a slow leaky integrator of pheromone-PN and
    contact activity (P1/pC1 persistent arousal, seconds timescale).
  * Motor: turning = DNa02/DNa01 left-right difference read out from the
    spiking network; pursuit gain and speed are gated by pC1 firing.
"""
from dataclasses import dataclass, field
import numpy as np
from brain import Brain

ARENA_R = 20.0          # mm
DT_BODY = 20.0          # ms per behavior step


@dataclass
class Fly:
    x: float
    y: float
    h: float            # heading, radians
    v: float = 0.0


@dataclass
class Trial:
    t: list = field(default_factory=list)
    male: list = field(default_factory=list)
    female: list = field(default_factory=list)
    rates: list = field(default_factory=list)
    arousal: list = field(default_factory=list)


READOUT = ["pher_ORN", "pher_PN", "contact_asc", "pC1", "LC10a_L", "LC10a_R", "DNa02_L", "DNa02_R"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Courtship:
    def __init__(self, silence=(), seed=0, duration_s=30.0):
        self.rng = np.random.default_rng(seed)
        self.brain = Brain(seed=seed)
        for g in silence:
            for s in ("_L", "_R"):
                if g + s in self.brain.groups:
                    self.brain.silence(g + s)
            if g in self.brain.groups:
                self.brain.silence(g)
        self.silence = set(silence)
        self.duration = duration_s
        r = self.rng
        a = r.uniform(0, 2 * np.pi)
        self.female = Fly(6 * np.cos(a), 6 * np.sin(a), r.uniform(0, 2 * np.pi))
        b = a + np.pi + r.uniform(-0.5, 0.5)
        self.male = Fly(12 * np.cos(b), 12 * np.sin(b), r.uniform(0, 2 * np.pi))
        self.arousal = 0.0

    # ---------------- world -> brain ----------------
    def senses(self):
        m, f = self.male, self.female
        dx, dy = f.x - m.x, f.y - m.y
        d = np.hypot(dx, dy)
        bearing = wrap(np.arctan2(dy, dx) - m.h)          # + = female on male's left
        drive = {}
        # pheromone at each antenna (0.3 mm lateral offsets)
        for side, sgn in (("L", 1), ("R", -1)):
            ax = m.x - sgn * 0.3 * np.sin(m.h)
            ay = m.y + sgn * 0.3 * np.cos(m.h)
            c = np.exp(-np.hypot(f.x - ax, f.y - ay) / 8.0)
            drive[f"pher_ORN_{side}"] = 150 * c
        touch = d < 2.5
        drive["contact_asc_L"] = drive["contact_asc_R"] = 150.0 if touch else 0.0
        # vision: angular size of a 2.5 mm fly, each eye sees ipsilateral 0..170 deg
        size = np.degrees(2 * np.arctan(1.25 / max(d, 0.5)))
        strength = min(1.0, size / 25.0)
        for side, sgn in (("L", 1), ("R", -1)):
            th = sgn * np.degrees(bearing)                 # angle into this eye's field
            in_field = -10 < th < 170
            drive[f"LC10a_{side}"] = 80 * strength if in_field else 0.0
        # pC1 driven by the arousal integrator
        drive["pC1_L"] = drive["pC1_R"] = 120 * self.arousal
        return drive, d, bearing, touch

    # ---------------- brain -> world ----------------
    def update_arousal(self, rates, touch):
        pn = self.brain.rate(rates, "pher_PN_L") + self.brain.rate(rates, "pher_PN_R")
        contact = self.brain.rate(rates, "contact_asc_L") + self.brain.rate(rates, "contact_asc_R")
        target = np.clip(pn / 250 + contact / 150, 0, 1)
        tau = 3000.0                                       # ms; P1 persistence
        self.arousal += (target - self.arousal) * DT_BODY / tau * (4 if target > self.arousal else 1)
        self.arousal = float(np.clip(self.arousal, 0, 1))

    def move(self, rates):
        b, m = self.brain, self.male
        left = b.rate(rates, "DNa02_L") + 0.5 * b.rate(rates, "DNa01_L")
        right = b.rate(rates, "DNa02_R") + 0.5 * b.rate(rates, "DNa01_R")
        pc1 = b.rate(rates, "pC1_L") + b.rate(rates, "pC1_R")
        gain = pc1 / (pc1 + 30.0)                          # arousal gating of pursuit
        dt = DT_BODY / 1000
        omega = 0.035 * gain * (left - right) + self.rng.normal(0, 1.2)  # rad/s
        m.h = wrap(m.h + omega * dt)
        m.v = 2.5 + 8.0 * gain                             # mm/s
        self._step(m, dt)

    def move_female(self):
        f, dt = self.female, DT_BODY / 1000
        if self.rng.random() < 0.01:
            f.v = 0.0 if f.v > 0 else self.rng.uniform(2, 5)   # walk / pause bouts
        f.h = wrap(f.h + self.rng.normal(0, 2.0) * np.sqrt(dt))
        self._step(f, dt)

    @staticmethod
    def _step(fly, dt):
        fly.x += fly.v * np.cos(fly.h) * dt
        fly.y += fly.v * np.sin(fly.h) * dt
        r = np.hypot(fly.x, fly.y)
        if r > ARENA_R:                                    # slide along the wall
            fly.x, fly.y = fly.x / r * ARENA_R, fly.y / r * ARENA_R
            fly.h = wrap(np.arctan2(fly.y, fly.x) + np.pi / 2 * np.sign(wrap(fly.h - np.arctan2(fly.y, fly.x))))

    def run(self):
        tr = Trial()
        self.female.v = 3.0
        for k in range(int(self.duration * 1000 / DT_BODY)):
            drive, d, bearing, touch = self.senses()
            rates = self.brain.run(drive, DT_BODY)
            self.update_arousal(rates, touch)
            self.move(rates)
            self.move_female()
            tr.t.append(k * DT_BODY / 1000)
            tr.male.append((self.male.x, self.male.y, self.male.h))
            tr.female.append((self.female.x, self.female.y, self.female.h))
            tr.rates.append([self.group_rate(rates, g) for g in READOUT])
            tr.arousal.append(self.arousal)
        for k in ("t", "male", "female", "rates", "arousal"):
            setattr(tr, k, np.asarray(getattr(tr, k)))
        return tr

    def group_rate(self, rates, g):
        gs = self.brain.groups
        if g in gs:
            return rates[gs[g]].mean()
        return rates[np.r_[gs[g + "_L"], gs[g + "_R"]]].mean()


def courtship_index(tr):
    """Fraction of time the male is following: within 5 mm and facing her (+-45 deg)."""
    m, f = tr.male, tr.female
    d = np.hypot(f[:, 0] - m[:, 0], f[:, 1] - m[:, 1])
    bearing = wrap(np.arctan2(f[:, 1] - m[:, 1], f[:, 0] - m[:, 0]) - m[:, 2])
    return float(np.mean((d < 5) & (np.abs(bearing) < np.pi / 4))), float(d.mean())
