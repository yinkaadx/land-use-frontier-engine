"""Land Use Frontier Engine. A multi objective spatial optimization laboratory for an agricultural frontier,
built as independent candidate work product by Yinka Aderibigbe for the doctoral position Spatial Optimization of
Land Use in Brazil (project Frontiers of MATOPIBA landscapes) at Utrecht University, Department of Human Geography
and Spatial Planning. The engine generates a synthetic frontier landscape in the spirit of the MATOPIBA region,
soy on the flat plateaus, cattle and degraded pasture below, cerrado woodland and grassland in between, gallery
forest along the drainage, and protected blocks. It then compares three ways of planning the next wave of
expansion: trend expansion by agricultural aptitude alone, weighted overlay ranking as practised in GIS, and a
population based multi objective search in the NSGA-II family that returns a Pareto frontier over production,
carbon, biodiversity and water. Frontiers are recomputed under a drier climate and a higher soy demand, and a
stakeholder panel shows how stated preferences translate, or fail to translate, into impact indicators and plans.
Everything is synthetic and illustrative; no real parcel, person or municipality is represented."""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Land Use Frontier Engine", page_icon="🌾", layout="wide")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98"

# land use classes
WOOD, GRASS, GAL, DEGR, PAST, SOY = 0, 1, 2, 3, 4, 5
CLASS_NAMES = ["cerrado woodland", "cerrado grassland", "gallery forest", "degraded pasture", "pasture", "soy cropland"]
CLASS_COLORS = ["#2e7d4f", "#a9c47f", "#14655a", "#d9b38c", "#c9a24b", "#eb6834"]
CARBON = np.array([62.0, 24.0, 92.0, 8.0, 11.0, 5.0])       # carbon stock proxy per cell
BIO = np.array([1.00, 0.78, 1.30, 0.12, 0.10, 0.05])         # habitat value per cell before endemism weighting
INFIL = np.array([1.00, 1.00, 1.00, 0.45, 0.42, 0.15])       # infiltration factor for recharge cells
H, W = 36, 54

def smooth(z, k):
    for _ in range(k):
        z = sum(np.roll(z, s, a) for s in (-1, 0, 1) for a in (0, 1) if not (s == 0 and a == 1)) / 5.0
    return z

@st.cache_data(show_spinner=False)
def make_world(seed):
    rng = np.random.default_rng(seed)
    gx = np.linspace(0, 1, W)[None, :] * np.ones((H, 1))
    elev = smooth(rng.normal(0, 1, (H, W)), 6) * 0.9 + gx * 1.4          # plateau rises to the east
    elev = (elev - elev.min()) / (elev.max() - elev.min())
    apt = np.clip(0.15 + 0.75 * elev + smooth(rng.normal(0, 0.5, (H, W)), 4), 0.05, 1.1)  # flat plateau farms best
    apt = apt / apt.max()
    rip = elev < np.quantile(elev, 0.10)                                  # drainage lines
    rech = elev > np.quantile(elev, 0.78)                                 # sandy plateau recharge
    rech_s = smooth(rech.astype(float), 3); rech_s = rech_s / rech_s.max()
    prot = np.zeros((H, W), bool); prot[3:11, 4:14] = True; prot[26:34, 34:44] = True
    rich = np.clip(0.55 + smooth(rng.normal(0, 1, (H, W)), 5) * 1.2, 0.25, 2.0)  # endemism field
    lu = np.full((H, W), GRASS)
    lu[smooth(rng.normal(0, 1, (H, W)), 5) > 0.05] = WOOD
    soy_seed = (apt > np.quantile(apt, 0.82)) & ~rip & ~prot              # existing soy on the best plateau
    lu[soy_seed] = SOY
    past_band = (apt > np.quantile(apt, 0.45)) & (lu != SOY) & ~rip & ~prot
    pr = rng.random((H, W))
    lu[past_band & (pr < 0.28)] = PAST
    lu[past_band & (pr > 0.86)] = DEGR
    lu[rip] = np.where(rng.random((H, W))[rip] < 0.70, GAL, DEGR)         # most gallery forest stands, a third was cleared
    lu[prot] = WOOD
    return dict(elev=elev, apt=apt, rip=rip, rech=rech, rech_s=rech_s, prot=prot, rich=rich, lu0=lu)

def candidates(world):
    lu0, prot, rip, apt = world["lu0"], world["prot"], world["rip"], world["apt"]
    soy = lu0 == SOY
    near = soy.copy()
    for _ in range(5):
        near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    nat_idx = np.flatnonzero((((lu0 == WOOD) | (lu0 == GRASS)) & ~prot & ~rip & near).ravel())
    deg_idx = np.flatnonzero((lu0 == DEGR).ravel())
    deg_rip = world["rip"].ravel()[deg_idx]                                # riparian degraded cells may only be kept or restored
    return nat_idx, deg_idx, deg_rip

def scen_params(scen):
    if scen == "drier":
        return dict(cmul_base=0.66, cmul_rech=0.22, adj=0.16, recyc=0.30)
    return dict(cmul_base=0.97, cmul_rech=0.06, adj=0.08, recyc=0.18)

def evaluate(nat_g, deg_g, world, scen):
    """nat_g (n, n_nat) in {0 keep, 1 soy}; deg_g (n, n_deg) in {0 keep, 1 soy, 2 restore}."""
    n = nat_g.shape[0]
    lu0, apt, rich, rip, rech, rech_s, prot = (world[k] for k in ("lu0", "apt", "rich", "rip", "rech", "rech_s", "prot"))
    nat_idx, deg_idx, deg_rip = candidates(world)
    p = scen_params(scen)
    lu = np.broadcast_to(lu0.ravel(), (n, H * W)).copy()
    restored = np.zeros((n, H * W), bool)
    lu[:, nat_idx] = np.where(nat_g == 1, SOY, lu[:, nat_idx])
    dd = lu[:, deg_idx]
    dd = np.where(deg_g == 1, SOY, dd)
    rest_to = np.where(deg_rip[None, :], GAL, GRASS)
    dd = np.where(deg_g == 2, rest_to, dd)
    lu[:, deg_idx] = dd
    restored[:, deg_idx] = deg_g == 2
    lu = lu.reshape(n, H, W); restored = restored.reshape(n, H, W)
    native = (lu == WOOD) | (lu == GRASS) | (lu == GAL)
    nn = sum(np.roll(native, s, a) for s in (1, -1) for a in (1, 2)).astype(float) / 4.0
    ns0 = ((lu0 == WOOD) | (lu0 == GRASS) | (lu0 == GAL)).mean()
    recyc = 1.0 - p["recyc"] * np.clip(1.0 - native.mean((1, 2)) / ns0, 0, None)   # rainfall recycling: clearing the cerrado dries the region
    cmul = p["cmul_base"] + p["cmul_rech"] * rech_s
    yields = apt[None] * cmul[None] * (1 + p["adj"] * nn) * recyc[:, None, None]
    P = (yields * (lu == SOY)).sum((1, 2)) + 0.25 * (apt[None] * (lu == PAST)).sum((1, 2)) * recyc + 0.05 * (apt[None] * (lu == DEGR)).sum((1, 2))
    young = np.where(restored, 0.5, 1.0)
    C = (CARBON[lu] * young).sum((1, 2))
    Bcell = BIO[lu] * rich[None] * np.where(restored, 0.6, 1.0)
    B = Bcell.sum((1, 2)) + 0.30 * (native * nn).sum((1, 2))
    Wr = (INFIL[lu] * rech[None]).sum((1, 2))
    Wg = (native & rip[None]).sum((1, 2)) / rip.sum()
    return P, C, B, Wr, Wg

def repair(nat_g, world, floor=0.35):
    """Legal reserve style repair: native share of unprotected land must stay above the floor."""
    nat_idx, _, _ = candidates(world)
    lu0, prot = world["lu0"], world["prot"]
    base_nat = (((lu0 == WOOD) | (lu0 == GRASS) | (lu0 == GAL)) & ~prot).sum()
    unprot = (~prot).sum()
    order = np.argsort(world["apt"].ravel()[nat_idx])                      # give back the worst land first
    for i in range(nat_g.shape[0]):
        over = int(np.floor(base_nat - floor * unprot))
        k = int(nat_g[i].sum())
        if k > over:
            drop = [j for j in order if nat_g[i, j] == 1][: k - over]
            nat_g[i, drop] = 0
    return nat_g

def norm_pack(P, C, B, Wr, Wg, base):
    Wq = 0.5 * (Wr / base["Wr"] + Wg / max(base["Wg"], 1e-9))
    return np.stack([P / base["P"], C / base["C"], B / base["B"], Wq], 1)

def nd_front(F):
    n = F.shape[0]; keep = np.ones(n, bool)
    for i in range(n):
        if keep[i]:
            dom = (F >= F[i]).all(1) & (F > F[i]).any(1)
            if dom.any():
                keep[i] = False
    return keep

def nsga2(world, scen, seed, base, target, gens=60, pop=48, extra=None):
    rng = np.random.default_rng(seed * 1000 + (7 if scen == "baseline" else 13))
    nat_idx, deg_idx, deg_rip = candidates(world)
    n_nat, n_deg = len(nat_idx), len(deg_idx)
    nat = (rng.random((pop, n_nat)) < 0.15).astype(np.int8)
    deg = rng.choice(3, (pop, n_deg), p=[0.6, 0.2, 0.2]).astype(np.int8)
    deg[:, :] = np.where(deg_rip[None, :] & (deg == 1), 0, deg)
    # interpretable reference plans join the initial population, disclosed in the method notes
    apt_rank = np.argsort(-world["apt"].ravel()[nat_idx])
    nat[0] = 0; deg[0] = 0                                                 # the current landscape
    tr_nat, tr_deg = trend_genes(world, scen, target)
    nat[1], deg[1] = tr_nat, tr_deg                                        # trend expansion
    nat[2] = 0; deg[2] = np.where(deg_rip, 0, 1)                           # degraded pasture first, no native cleared
    nat[3] = 0; deg[3] = 2                                                 # full restoration corner
    for si, dens in zip((4, 5, 6, 7, 8, 9), (0.10, 0.25, 0.45, 0.65, 0.85, 1.0)):
        nat[si] = 0; nat[si, apt_rank[: int(dens * n_nat)]] = 1
        deg[si] = np.where(deg_rip, 0, 1)                                  # expansion plus full intensification
    rech_on_nat = world["rech"].ravel()[nat_idx]
    nat[10] = 1; deg[10] = np.where(deg_rip, 2, 1)                         # riparian restoration beside full expansion
    nat[11] = np.where(rech_on_nat, 0, 1); deg[11] = np.where(deg_rip, 0, 1)  # plateau recharge guardian
    if extra:
        for k, (en, ed_) in enumerate(extra):
            if 12 + k < pop:
                nat[12 + k], deg[12 + k] = en, ed_
    nat = repair(nat, world)
    P, C, B, Wr, Wg = evaluate(nat, deg, world, scen)
    F = norm_pack(P, C, B, Wr, Wg, base)
    for g in range(gens):
        ranks, crowd = rank_crowd(F)
        idx = tournament(rng, ranks, crowd, pop)
        cn, cd = nat[idx], deg[idx]
        half = pop // 2
        m = rng.random((half, n_nat)) < 0.5
        c1n = np.where(m, cn[:half], cn[half:]); c2n = np.where(m, cn[half:], cn[:half])
        md = rng.random((half, n_deg)) < 0.5
        c1d = np.where(md, cd[:half], cd[half:]); c2d = np.where(md, cd[half:], cd[:half])
        on = np.vstack([c1n, c2n]); od = np.vstack([c1d, c2d])
        mut = rng.random(on.shape) < (2.0 / n_nat)
        on = np.where(mut, 1 - on, on)
        mutd = rng.random(od.shape) < (2.0 / n_deg)
        od = np.where(mutd, rng.choice(3, od.shape).astype(np.int8), od)
        od = np.where(deg_rip[None, :] & (od == 1), 0, od)
        on = repair(on.astype(np.int8), world)
        Po, Co, Bo, Wro, Wgo = evaluate(on, od, world, scen)
        Fo = norm_pack(Po, Co, Bo, Wro, Wgo, base)
        nat = np.vstack([nat, on]); deg = np.vstack([deg, od]); F = np.vstack([F, Fo])
        raw = np.vstack([np.stack([P, C, B, Wr, Wg], 1), np.stack([Po, Co, Bo, Wro, Wgo], 1)])
        ranks, crowd = rank_crowd(F)
        order = np.lexsort((-crowd, ranks))[:pop]
        nat, deg, F = nat[order], deg[order], F[order]
        P, C, B, Wr, Wg = raw[order].T
    keep = nd_front(F)
    return dict(nat=nat[keep], deg=deg[keep], F=F[keep], raw=np.stack([P, C, B, Wr, Wg], 1)[keep], base=base)

def rank_crowd(F):
    n = F.shape[0]; ranks = np.zeros(n, int); left = np.ones(n, bool); r = 0
    while left.any():
        sub = np.flatnonzero(left)
        keep = nd_front(F[sub])
        ranks[sub[keep]] = r; left[sub[keep]] = False; r += 1
    crowd = np.zeros(n)
    for m in range(F.shape[1]):
        o = np.argsort(F[:, m]); span = F[o[-1], m] - F[o[0], m] + 1e-12
        crowd[o[0]] = crowd[o[-1]] = np.inf
        crowd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / span
    return ranks, crowd

def tournament(rng, ranks, crowd, k):
    a, b = rng.integers(0, len(ranks), k), rng.integers(0, len(ranks), k)
    better = (ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) & (crowd[a] >= crowd[b]))
    return np.where(better, a, b)

def baseline_scores(world, scen):
    nat_idx, deg_idx, _ = candidates(world)
    z = np.zeros((1, len(nat_idx)), np.int8); zd = np.zeros((1, len(deg_idx)), np.int8)
    P, C, B, Wr, Wg = evaluate(z, zd, world, scen)
    return dict(P=P[0], C=C[0], B=B[0], Wr=Wr[0], Wg=Wg[0])

def trend_genes(world, scen, target):
    """Historical pattern: expand over native cerrado in aptitude order, leave the degraded pasture alone."""
    nat_idx, deg_idx, _ = candidates(world)
    order = np.argsort(-world["apt"].ravel()[nat_idx])
    nat = np.zeros((1, len(nat_idx)), np.int8); deg = np.zeros((1, len(deg_idx)), np.int8)
    for j in order:
        nat[0, j] = 1
        P, *_ = evaluate(nat, deg, world, scen)
        if P[0] >= target:
            break
    return nat[0], deg[0]

def overlay_points(world, scen, weights, base):
    """Weighted overlay ranking, the common GIS practice: score every candidate change once against the current
    landscape, apply every change whose weighted score is positive, then measure what actually results."""
    nat_idx, deg_idx, deg_rip = candidates(world)
    outs, genes = [], []
    for w in weights:
        nat = np.zeros((1, len(nat_idx)), np.int8); deg = np.zeros((1, len(deg_idx)), np.int8)
        gains_nat, gains_deg_soy, gains_deg_rest = unit_gains(world, scen, base, w)
        nat[0] = (gains_nat > 0).astype(np.int8)
        pick_rest = (gains_deg_rest > 0) & (gains_deg_rest >= gains_deg_soy)
        pick_soy = (gains_deg_soy > 0) & ~pick_rest & ~deg_rip
        deg[0] = np.where(pick_rest, 2, np.where(pick_soy, 1, 0)).astype(np.int8)
        nat = repair(nat, world)
        P, C, B, Wr, Wg = evaluate(nat, deg, world, scen)
        outs.append(norm_pack(P, C, B, Wr, Wg, base)[0])
        genes.append((nat[0].copy(), deg[0].copy()))
    return np.array(outs), genes

def unit_gains(world, scen, base, w):
    p = scen_params(scen)
    apt, rich, rech, rip, lu0 = world["apt"].ravel(), world["rich"].ravel(), world["rech"].ravel(), world["rip"].ravel(), world["lu0"].ravel()
    nat_idx, deg_idx, deg_rip = candidates(world)
    cmul = (p["cmul_base"] + p["cmul_rech"] * world["rech_s"]).ravel()
    dP_nat = apt[nat_idx] * cmul[nat_idx]
    dC_nat = -(CARBON[lu0[nat_idx]] - CARBON[SOY])
    dB_nat = -(BIO[lu0[nat_idx]] - BIO[SOY]) * rich[nat_idx]
    dW_nat = -(INFIL[lu0[nat_idx]] - INFIL[SOY]) * rech[nat_idx] / base["Wr"] * 0.5
    dP_ds = apt[deg_idx] * cmul[deg_idx] - 0.05 * apt[deg_idx]
    dC_ds = -(CARBON[DEGR] - CARBON[SOY]) * np.ones(len(deg_idx))
    dB_ds = -(BIO[DEGR] - BIO[SOY]) * rich[deg_idx]
    dW_ds = -(INFIL[DEGR] - INFIL[SOY]) * rech[deg_idx] / base["Wr"] * 0.5
    dP_dr = -0.05 * apt[deg_idx]
    dC_dr = (np.where(deg_rip, CARBON[GAL], CARBON[GRASS]) * 0.5 - CARBON[DEGR])
    dB_dr = (np.where(deg_rip, BIO[GAL], BIO[GRASS]) * 0.6 - BIO[DEGR]) * rich[deg_idx]
    dW_dr = ((1.0 - INFIL[DEGR]) * rech[deg_idx] / base["Wr"] * 0.5) + np.where(deg_rip, 0.5 / max(world["rip"].sum(), 1), 0) / max(base["Wg"], 1e-9) * 0.5
    sc = lambda dP, dC, dB, dW: w[0] * dP / base["P"] + w[1] * dC / base["C"] + w[2] * dB / base["B"] + w[3] * dW
    return sc(dP_nat, dC_nat, dB_nat, dW_nat), sc(dP_ds, dC_ds, dB_ds, dW_ds), sc(dP_dr, dC_dr, dB_dr, dW_dr)

@st.cache_data(show_spinner=False)
def run_all(seed, gens):
    world = make_world(seed)
    base0 = baseline_scores(world, "baseline")                             # one currency: today's landscape under today's climate
    target = 1.5 * base0["P"]
    out = {}
    for scen in ("baseline", "drier"):
        Wsimplex = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0.5, 0.5, 0, 0), (0.5, 0, 0.5, 0), (0.5, 0, 0, 0.5),
                    (0, 0.5, 0.5, 0), (0, 0.5, 0, 0.5), (0, 0, 0.5, 0.5), (0.34, 0.22, 0.22, 0.22), (0.6, 0.2, 0.1, 0.1), (0.25, 0.25, 0.25, 0.25)]
        ov, ov_genes = overlay_points(world, scen, Wsimplex, base0)
        res = nsga2(world, scen, seed, base0, target, gens=gens, extra=ov_genes)
        tr_nat, tr_deg = trend_genes(world, scen, target)
        Pt, Ct, Bt, Wrt, Wgt = evaluate(tr_nat[None, :], tr_deg[None, :], world, scen)
        trF = norm_pack(Pt, Ct, Bt, Wrt, Wgt, base0)[0]
        out[scen] = dict(res=res, trend=trF, trend_genes=(tr_nat, tr_deg), overlay=ov)
    return world, out

def genes_to_change(world, nat_g, deg_g):
    nat_idx, deg_idx, deg_rip = candidates(world)
    lu0 = world["lu0"]
    change = np.zeros(H * W)                                              # 0 kept native, 1 farmland already, 2 new soy on native, 3 new soy on degraded, 4 restored
    lu0r = lu0.ravel()
    change[(lu0r == SOY) | (lu0r == PAST) | (lu0r == DEGR)] = 1
    change[nat_idx[nat_g == 1]] = 2
    change[deg_idx[deg_g == 1]] = 3
    change[deg_idx[deg_g == 2]] = 4
    return change.reshape(H, W)

def map_df(world, arr):
    ys, xs = np.mgrid[0:H, 0:W]
    return pd.DataFrame({"x": xs.ravel(), "y": ys.ravel(), "v": np.asarray(arr).ravel()})

def cat_map(world, arr, names, colors, title):
    df = map_df(world, arr); df["lab"] = [names[int(v)] for v in df["v"]]
    return alt.Chart(df).mark_rect().encode(
        x=alt.X("x:O", axis=None), y=alt.Y("y:O", axis=None),
        color=alt.Color("lab:N", scale=alt.Scale(domain=names, range=colors), legend=alt.Legend(title=None, orient="bottom", columns=2, labelLimit=220)),
    ).properties(width=alt.Step(6.2), height=alt.Step(6.2), title=title)

def num_map(world, arr, scheme, title):
    return alt.Chart(map_df(world, arr)).mark_rect().encode(
        x=alt.X("x:O", axis=None), y=alt.Y("y:O", axis=None),
        color=alt.Color("v:Q", scale=alt.Scale(scheme=scheme), legend=alt.Legend(title=None, orient="bottom", gradientLength=140)),
    ).properties(width=alt.Step(6.2), height=alt.Step(6.2), title=title)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("**Controls**")
    seed = st.slider("world and search seed", 1, 12, 7)
    gens = st.slider("generations of search", 30, 120, 60, 10)
    st.caption(
        "Every number and claim on the page is recomputed from the run you configure here; nothing is hard coded. "
        "The default landscape is 36 by 54 cells and the search runs a population of 48 plans."
    )

world, OUT = run_all(seed, gens)
lu0 = world["lu0"]
shares = {CLASS_NAMES[k]: float((lu0 == k).mean()) for k in range(6)}
nat_idx, deg_idx, _ = candidates(world)

st.title("Land Use Frontier Engine")
st.caption(
    "MATOPIBA, the cerrado frontier across Maranhao, Tocantins, Piaui and Bahia, is where Brazilian soy expands and where "
    "carbon, biodiversity and water are lost or kept. A plan for such a region is not one map but a set of trade offs, and the "
    "honest object to compute is the Pareto frontier: the plans that cannot be improved on one objective without paying on "
    "another. This engine builds a synthetic frontier landscape, then compares trend expansion, weighted overlay ranking and a "
    "multi objective evolutionary search, recomputes the frontier under a drier climate and a higher demand, and translates "
    "stakeholder preferences into impact indicators and back into plans. Independent candidate work product by Yinka Aderibigbe "
    "for the position Spatial Optimization of Land Use in Brazil, Utrecht University; everything on this page is synthetic and "
    "illustrative, and no real parcel or person is represented."
)

# ---------------------------------------------------------------- section 1
st.subheader("1. A synthetic frontier landscape")
c1, c2, c3 = st.columns(3)
with c1:
    st.altair_chart(cat_map(world, lu0, CLASS_NAMES, CLASS_COLORS, "land use today"), use_container_width=False)
with c2:
    st.altair_chart(num_map(world, world["apt"], "yelloworangebrown", "agricultural aptitude"), use_container_width=False)
with c3:
    st.altair_chart(num_map(world, BIO[lu0] * world["rich"], "greens", "habitat value"), use_container_width=False)
st.caption(
    f"The generator follows the geography that makes MATOPIBA contested: existing soy ({shares['soy cropland']:.0%} of cells) sits on the "
    f"flat, high aptitude plateau; pasture ({shares['pasture']:.0%}) and degraded pasture ({shares['degraded pasture']:.0%}) sit below it; cerrado woodland and "
    f"grassland ({shares['cerrado woodland'] + shares['cerrado grassland']:.0%} together) fill the rest; gallery forest ({shares['gallery forest']:.0%}) lines the drainage and carries the "
    f"highest carbon and habitat values; two blocks are protected. Recharge cells cluster on the sandy plateau, so the water "
    f"objective rewards keeping infiltration there; a legal reserve rule in the spirit of the Forest Code keeps native cover above "
    f"35 percent of unprotected land in every plan. The decision space holds {len(nat_idx)} native cells in the expansion band, each kept "
    f"or converted to soy, and {len(deg_idx)} degraded pasture cells, each kept, intensified to soy, or restored toward native cover."
)

# ---------------------------------------------------------------- section 2
st.subheader("2. Three ways to plan the same expansion")
scen = "baseline"
R = OUT[scen]["res"]; trF = OUT[scen]["trend"]; ov = OUT[scen]["overlay"]
F = R["F"]
front_df = pd.DataFrame(F, columns=["P", "C", "B", "Wq"]); front_df["kind"] = "Pareto frontier plan"
ov_df = pd.DataFrame(ov, columns=["P", "C", "B", "Wq"]); ov_df["kind"] = "weighted overlay ranking"
tr_df = pd.DataFrame([trF], columns=["P", "C", "B", "Wq"]); tr_df["kind"] = "trend expansion"
cur = R["base"]
cur_df = pd.DataFrame([norm_pack(np.array([cur["P"]]), np.array([cur["C"]]), np.array([cur["B"]]), np.array([cur["Wr"]]), np.array([cur["Wg"]]), cur)[0]], columns=["P", "C", "B", "Wq"]); cur_df["kind"] = "current landscape"
allpts = pd.concat([front_df, ov_df, tr_df, cur_df]).reset_index(drop=True)
long = allpts.melt(id_vars=["P", "kind"], value_vars=["C", "B", "Wq"], var_name="objective", value_name="value")
long["objective"] = long["objective"].map({"C": "carbon stock", "B": "biodiversity", "Wq": "water services"})
ch = alt.Chart(long).mark_point(filled=True, size=70).encode(
    x=alt.X("P:Q", title="soy and cattle production", scale=alt.Scale(zero=False)),
    y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
    color=alt.Color("kind:N", scale=alt.Scale(domain=["Pareto frontier plan", "weighted overlay ranking", "trend expansion", "current landscape"],
                                              range=[BLUE, GREY, ORANGE, AQUA]), legend=alt.Legend(orient="bottom", title=None)),
    shape=alt.Shape("kind:N", scale=alt.Scale(domain=["Pareto frontier plan", "weighted overlay ranking", "trend expansion", "current landscape"],
                                              range=["circle", "cross", "triangle-up", "square"]), legend=None),
    tooltip=[alt.Tooltip("P:Q", format=".0f"), alt.Tooltip("value:Q", format=".2f"), "kind:N"],
).properties(width=300, height=260).facet(column=alt.Column("objective:N", title=None)).resolve_scale(y="independent")
st.altair_chart(ch, use_container_width=False)

# claims computed from the run
ge = F[:, 0] >= trF[0] - 1e-9
match = None
if ge.any():
    match = int(np.argmax(F[:, 1] * ge - 1e9 * (~ge)))
dom_by_front = np.array([((F >= o).all(1) & (F > o).any(1)).any() for o in ov])
n_distinct = len(np.unique(ov.round(3), axis=0))
claim2 = (
    f"With the seed and budget set in the sidebar, the search returns {len(F)} mutually nondominated plans. Trend expansion, aptitude "
    f"first over native cerrado as the region has historically expanded, reaches production {trF[0]:.2f} times today's while carbon "
    f"falls to {trF[1]:.2f} of today, biodiversity to {trF[2]:.2f} and water services to {trF[3]:.2f}. "
)
if match is not None:
    n_tr_conv = int(OUT[scen]["trend_genes"][0].sum())
    n_m_conv = int(R["nat"][match].sum())
    claim2 += (
        f"The frontier holds a plan at production {F[match, 0]:.2f} that keeps carbon at {F[match, 1]:.2f}, biodiversity at {F[match, 2]:.2f} and water "
        f"services at {F[match, 3]:.2f}, because it routes expansion through the degraded pasture, intensifying "
        f"{int((R['deg'][match] == 1).sum())} of {len(deg_idx)} degraded cells and clearing {n_m_conv} native cells where the trend clears {n_tr_conv}. "
    )
hiP = float(max(F[:, 0].max(), ov[:, 0].max()))
ov_u = np.unique(ov.round(3), axis=0)
ov_mid = int(((ov_u[:, 0] > 1.10) & (ov_u[:, 0] < hiP - 0.10)).sum())
fr_mid = int(((F[:, 0] > 1.10) & (F[:, 0] < hiP - 0.10)).sum())
claim2 += (
    f"Weighted overlay ranking, scoring each cell once against today's landscape and applying every positive change, was run for 13 "
    f"weight vectors: it returned only {n_distinct} distinct outcomes"
    + (f", {int(dom_by_front.sum())} of them strictly dominated by frontier plans" if dom_by_front.sum() else "")
    + f", and in the production band between 1.10 and {hiP - 0.10:.2f} times today, where negotiation between production and conservation "
    f"actually happens, it offers {ov_mid} plans where the frontier holds {fr_mid}. One pass scores also never see the two landscape "
    f"feedbacks, adjacency and rainfall recycling, that the search prices explicitly."
)
st.caption(claim2)

if match is not None:
    ch_names = ["kept native", "farmland already", "new soy on native", "new soy on degraded pasture", "restored"]
    ch_colors = ["#bfe0c8", "#efe5d4", ORANGE, "#b65718", AQUA]
    m1, m2 = st.columns(2)
    with m1:
        st.altair_chart(cat_map(world, genes_to_change(world, *OUT[scen]["trend_genes"]), ch_names, ch_colors, "trend expansion plan"), use_container_width=False)
    with m2:
        st.altair_chart(cat_map(world, genes_to_change(world, R["nat"][match], R["deg"][match]), ch_names, ch_colors, "frontier plan at matched production"), use_container_width=False)
    st.caption(
        "Same production, different geographies. Trend expansion clears the contiguous native block nearest the existing soy; the "
        "matched frontier plan spreads onto already degraded land and buys back connectivity by restoring cells the trend never "
        "touches. This is the pattern the Cerrado literature keeps finding: the region can grow output on land already cleared."
    )

# ---------------------------------------------------------------- section 3
st.subheader("3. Frontiers move when the future does")
rows = []
for scen_k, label in (("baseline", "baseline climate"), ("drier", "drier 2040s climate")):
    Fk = OUT[scen_k]["res"]["F"]
    E = Fk[:, 1:].mean(1)
    for i in range(len(Fk)):
        rows.append(dict(P=Fk[i, 0], E=E[i], scenario=label))
sc_df = pd.DataFrame(rows)
sc_df["view"] = "environment index (carbon, biodiversity, water; today = 1) against production"
xd = [float(sc_df["P"].min()) - 0.05, float(sc_df["P"].max()) + 0.05]
yd = [float(sc_df["E"].min()) - 0.02, float(sc_df["E"].max()) + 0.02]
ch3 = alt.Chart(sc_df).mark_point(filled=True, size=60).encode(
    x=alt.X("P:Q", title="production (times today)", scale=alt.Scale(domain=xd, nice=False)),
    y=alt.Y("E:Q", title=None, scale=alt.Scale(domain=yd, nice=False)),
    color=alt.Color("scenario:N", scale=alt.Scale(domain=["baseline climate", "drier 2040s climate"], range=[BLUE, ORANGE]), legend=alt.Legend(orient="bottom", title=None)),
).properties(width=620, height=300).facet(column=alt.Column("view:N", title=None))
st.altair_chart(ch3, use_container_width=False)
Fb, Fd = OUT["baseline"]["res"]["F"], OUT["drier"]["res"]["F"]
Eb, Ed = Fb[:, 1:].mean(1), Fd[:, 1:].mean(1)
maxPb, maxPd = Fb[:, 0].max(), Fd[:, 0].max()
floor = 0.93
pb = Fb[Eb >= floor, 0].max() if (Eb >= floor).any() else np.nan
pd_ = Fd[Ed >= floor, 0].max() if (Ed >= floor).any() else np.nan
dem = 1.5
eb = Eb[Fb[:, 0] >= dem].max() if (Fb[:, 0] >= dem).any() else np.nan
ed = Ed[Fd[:, 0] >= dem].max() if (Fd[:, 0] >= dem).any() else np.nan
c3txt = (
    f"The frontier is not a fact about the region; it is a fact about the region under a future. In the drier climate, yields fall "
    f"hardest away from the recharge belt, the microclimate value of neighbouring native vegetation doubles, and the rainfall "
    f"recycling penalty for clearing deepens, so the same search against the same landscape returns a different curve: attainable "
    f"production tops out at {maxPd:.2f} times today against {maxPb:.2f} under the baseline climate. "
)
if not (np.isnan(pb) or np.isnan(pd_)):
    c3txt += (
        f"Holding the environment index at {floor:.2f} of today, the production the region can offer falls from {pb:.2f} to {pd_:.2f} times today. "
    )
if not np.isnan(eb):
    c3txt += (
        f"Meeting a demand of 1.5 times today costs an environment index of {1 - eb:.2f} under the baseline"
        + (f" and {1 - ed:.2f} under the drier climate. " if not np.isnan(ed) else ", and under the drier climate no plan in the returned set reaches that demand at all. ")
    )
c3txt += (
    "A policy brief that publishes one optimal map hides all of this; publishing the frontier under named scenarios is what makes "
    "the trade offs, and their movement, discussable."
)
st.caption(c3txt)

# ---------------------------------------------------------------- section 4
st.subheader("4. From stated preferences to indicators to plans")
profiles = {
    "grain producers": np.array([0.70, 0.10, 0.05, 0.15]),
    "conservation agencies": np.array([0.05, 0.30, 0.50, 0.15]),
    "traditional communities": np.array([0.35, 0.05, 0.25, 0.35]),
    "water utility": np.array([0.15, 0.05, 0.20, 0.60]),
}
Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
raw = R["raw"]
picks = {}
for name, wgt in profiles.items():
    picks[name] = int(np.argmax(Fn @ wgt))
rows = []
for name, j in picks.items():
    rows.append(dict(panel=name, production=F[j, 0], carbon=F[j, 1], biodiversity=F[j, 2], water=F[j, 3]))
st.dataframe(pd.DataFrame(rows).set_index("panel").round(2), use_container_width=False)
# translation ambiguity: the same stated concern, water, read as recharge or as gallery forest integrity.
# the actor is a basin committee, which must keep the economy and the water running at once.
base = R["base"]
Wr_n = raw[:, 3] / base["Wr"]; Wg_n = raw[:, 4] / max(base["Wg"], 1e-9)
w_basin = np.array([0.35, 0.05, 0.05, 0.55])
score_rech = Fn[:, :3] @ w_basin[:3] + w_basin[3] * (Wr_n - Wr_n.min()) / (Wr_n.max() - Wr_n.min() + 1e-12)
score_gal = Fn[:, :3] @ w_basin[:3] + w_basin[3] * (Wg_n - Wg_n.min()) / (Wg_n.max() - Wg_n.min() + 1e-12)
j_rech, j_gal = int(np.argmax(score_rech)), int(np.argmax(score_gal))
m_rech = genes_to_change(world, R["nat"][j_rech], R["deg"][j_rech])
m_gal = genes_to_change(world, R["nat"][j_gal], R["deg"][j_gal])
diff_cells = float((m_rech != m_gal).mean())
p1, p2 = st.columns(2)
ch_names = ["kept native", "farmland already", "new soy on native", "new soy on degraded pasture", "restored"]
ch_colors = ["#bfe0c8", "#efe5d4", ORANGE, "#b65718", AQUA]
with p1:
    st.altair_chart(cat_map(world, m_rech, ch_names, ch_colors, "basin committee's water read as plateau recharge"), use_container_width=False)
with p2:
    st.altair_chart(cat_map(world, m_gal, ch_names, ch_colors, "basin committee's water read as gallery forest"), use_container_width=False)
n_picks = len(set(picks.values()))
same3 = picks["conservation agencies"] == picks["traditional communities"] == picks["water utility"]
if same3:
    agree_txt = (
        f"the producers' pick carries production {F[picks['grain producers'], 0]:.2f} at biodiversity {F[picks['grain producers'], 2]:.2f}, while the three other panels, whose stated "
        f"priorities differ, converge on the same restoration heavy plan at production {F[picks['conservation agencies'], 0]:.2f} and biodiversity {F[picks['conservation agencies'], 2]:.2f}, so the live "
        f"disagreement in this landscape is not among conservation minded actors but between them and the producers. "
    )
else:
    agree_txt = (
        f"the producers' pick carries production {F[picks['grain producers'], 0]:.2f} at biodiversity {F[picks['grain producers'], 2]:.2f} against the conservation agencies' "
        f"{F[picks['conservation agencies'], 0]:.2f} and {F[picks['conservation agencies'], 2]:.2f}, with the communities and the water utility between them. "
    )
st.caption(
    f"The four panels weigh the same indicators differently and select {n_picks} distinct plans from the same frontier, and the pattern of agreement is itself a "
    f"finding: " + agree_txt + "The subtler and, "
    f"for this project, more important point sits in the pair of maps above. A basin committee "
    f"that must keep the economy and the water running weighs production at 0.35 and water at 0.55, and its stated concern for "
    f"water was translated two defensible ways, as plateau recharge protection and as gallery forest integrity. The two translations "
    f"select plans that differ on {diff_cells:.0%} of all cells, production {F[j_rech, 0]:.2f} against {F[j_gal, 0]:.2f}, recharge served {Wr_n[j_rech]:.2f} against {Wr_n[j_gal]:.2f}, "
    f"gallery forest {Wg_n[j_rech]:.2f} against {Wg_n[j_gal]:.2f}. The step from what stakeholders say to what the algorithm counts is itself a modelling "
    f"decision, it is invisible in the final map, and it is why indicator translation belongs in the field work, not in the code alone."
)

# ---------------------------------------------------------------- section 5
st.subheader("5. What this engine is, and what it is not")
st.markdown(
    "**Honest scope.** The landscape, yields, carbon, habitat and infiltration values are synthetic and illustrative; the engine is a "
    "conceptual instrument about method, not a claim about any real municipality. What carries to the real project: the value of "
    "computing frontiers rather than single optima; trend expansion and weighted overlay as the baselines a frontier must beat and "
    "be compared against; scenario conditioned frontiers as the honest way to present futures; repair operators for legal reserve "
    "style constraints; and indicator translation treated as an empirical question for stakeholder work in Brazil rather than a "
    "settled input. What does not carry: every number. In the real project the land use map, yields and carbon would come from "
    "MapBiomas, Embrapa and the literature, the biodiversity layer from occurrence models with their own documented biases, the "
    "preferences from the field work this project shares with its sister PhD on co production of land use futures, and the "
    "optimizer would face uncertainty analysis in the tradition of the host group, frontier bands rather than frontier lines."
)
st.caption(
    "Method notes. The search is an NSGA-II style evolutionary algorithm, fast nondominated sorting with crowding distance, "
    "uniform crossover and per gene mutation, population 48, warm started with interpretable reference plans, the current "
    "landscape, trend expansion, degraded pasture first, full restoration, aptitude ranked expansion at six densities, two water "
    "sensitive variants, and the 13 weighted overlay plans themselves, all disclosed here and all subject to the same selection, "
    "so the returned frontier can only match or improve on the practice it is compared against. A repair operator enforces the "
    "native cover floor. Yields carry two landscape "
    "feedbacks, a local adjacency benefit from neighbouring native vegetation and a regional rainfall recycling penalty that "
    "grows with total clearing; the weighted overlay baseline scores each candidate change once against the current landscape, "
    "which is exactly why those interactions escape it. Code and tests are in the repository. Built with numpy, pandas, Altair "
    "and Streamlit."
)
