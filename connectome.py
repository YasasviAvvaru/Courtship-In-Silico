"""Extract a courtship subcircuit from the FlyWire v783 whole-brain connectome.

Keeps every neuron that lies on a short path (<= MAX_HOPS synapses, each edge
>= MIN_SYN synapses) between a sensory input group and an integrator/motor
group. Output: data/circuit.npz with a signed weight matrix and named groups.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).parent / "data"
MIN_SYN = 5

# (source group, target group, max path length in synapses)
PATHS = [
    ("pher_ORN", "pher_PN", 1),   # volatile pheromone receptors -> projection neurons
    ("pher_PN", "pC1", 3),        # projection neurons -> arousal hub (lateral horn etc.)
    ("contact_asc", "pC1", 3),    # contact chemosensory (ascending fru+) -> arousal hub
    ("LC10a", "DN_steer", 2),     # visual target -> steering descending neurons
    ("pC1", "DN_steer", 3),       # arousal hub -> steering (whatever the wiring offers)
]


def load():
    ann = pd.read_csv(DATA / "annotations.tsv", sep="\t", low_memory=False)
    comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(DATA / "Connectivity_783.parquet")
    N = len(comp)
    W = sp.csr_matrix(
        (con["Excitatory x Connectivity"].to_numpy(np.float32),
         (con.Presynaptic_Index.to_numpy(), con.Postsynaptic_Index.to_numpy())),
        shape=(N, N))
    return ann, comp, W


def group_defs(ann):
    ct = ann.cell_type.astype(str)
    return {
        # Or67d (DA1, cVA) and Or47b (VA1v, fly fatty-acid pheromones)
        "pher_ORN": ct.isin(["ORN_DA1", "ORN_VA1v"]),
        "pher_PN": ct.str.match(r"^(DA1|VA1v)_.*PN$"),
        "contact_asc": (ann.super_class == "ascending") & (ann.fru_dsx == "fru"),
        "LC10a": ct == "LC10a",
        "pC1": ct.str.startswith("pC1"),
        "DN_steer": ct.isin(["DNa02", "DNa01"]),
    }


def reach(src, M, k):
    """Hop distance from src along M (rows=pre). Returns int array, -1 = unreachable."""
    N = M.shape[0]
    dist = np.full(N, -1, np.int32)
    dist[src] = 0
    frontier = np.zeros(N, np.float32)
    frontier[src] = 1
    for h in range(1, k + 1):
        nxt = (M.T @ frontier) > 0
        nxt &= dist < 0
        dist[nxt] = h
        frontier = nxt.astype(np.float32)
    return dist


def main():
    ann, comp, W = load()
    N = W.shape[0]
    pos = pd.Series(np.arange(N), index=comp.index)
    ann = ann[ann.root_id.isin(comp.index)].copy()
    ann["idx"] = pos.loc[ann.root_id].to_numpy()

    strong = W.copy()
    strong.data = (np.abs(strong.data) >= MIN_SYN).astype(np.float32)
    strong.eliminate_zeros()

    gmask = group_defs(ann)
    gidx = {g: ann.loc[m, "idx"].to_numpy() for g, m in gmask.items()}

    keep = np.zeros(N, bool)
    for s, t, L in PATHS:
        d_fwd = reach(gidx[s], strong, L)
        d_bwd = reach(gidx[t], strong.T.tocsr(), L)
        on_path = (d_fwd >= 0) & (d_bwd >= 0) & (d_fwd + d_bwd <= L)
        print(f"{s:>12} -> {t:<9} (<= {L} hops): {on_path.sum():5d} neurons on paths")
        keep |= on_path
    for g in gidx.values():
        keep[g] = True

    sub = np.flatnonzero(keep)
    Wsub = W[sub][:, sub].toarray().astype(np.float32)  # rows = pre, cols = post
    local = {old: i for i, old in enumerate(sub)}
    info = ann.set_index("idx").reindex(sub)

    groups = {}
    for g, ix in gidx.items():
        sides = info.loc[ix, "side"].to_numpy()
        li = np.array([local[i] for i in ix])
        groups[f"{g}_L"] = li[sides == "left"]
        groups[f"{g}_R"] = li[sides == "right"]
    for name in ["DNa02", "DNa01"]:
        for side, tag in [("left", "L"), ("right", "R")]:
            m = (info.cell_type == name).to_numpy() & (info.side == side).to_numpy()
            groups[f"{name}_{tag}"] = np.flatnonzero(m)

    print(f"subcircuit: {len(sub)} neurons, {(Wsub != 0).sum()} connections")
    print({k: len(v) for k, v in groups.items()})
    np.savez_compressed(
        DATA / "circuit.npz", W=Wsub, root_ids=comp.index.to_numpy()[sub],
        cell_type=info.cell_type.astype(str).to_numpy(),
        super_class=info.super_class.astype(str).to_numpy(),
        side=info.side.astype(str).to_numpy(),
        **{f"g_{k}": v for k, v in groups.items()})


if __name__ == "__main__":
    main()
