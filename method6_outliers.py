"""Method 6 (Graph Signal Recovery): flag outliers, then refill them from their neighbours.

Main functions: flag_outliers, refill_outliers, solve_two_stage.
Defaults (k = 2.5) are the values used in the group result; no tuning needed.
"""
import numpy as np
import networkx as nx


def weighted_median(values, weights):
    """Weighted median of a list of values.

    Input:  values (array, length d), weights (array, length d, positive).
    Output: float, the first sorted value whose running weight reaches half of the total.
    """
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w)
    return v[np.searchsorted(c, 0.5 * c[-1])]


def robust_spread(v):
    """Robust standard deviation: 1.4826 * MAD.

    Input:  v (array).
    Output: float.
    """
    v = np.asarray(v, dtype=float)
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def detect_outliers(G, y, mask=None, k=2.5, s_min=0.0, s_fixed=None):
    """One pass of Stage 1: flag nodes that disagree with the weighted median of their neighbours.

    Input:
        G       networkx graph, nodes 0..n-1, edge attribute "weight".
        y       float array, length n.
        mask    bool array, True = observed (default: all observed). Hidden nodes are not used or flagged.
        k       threshold: flag if |r_i - median(r)| > k * s.
        s_min   lower bound for s.
        s_fixed use this s instead of estimating it.
    Output: (flagged, r, s)
        flagged bool array, length n, True = flagged.
        r       float array, length n, residual y_i - neighbour median (NaN if no observed neighbour).
        s       float, robust scale used.
    """
    n = len(y)
    obs = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    m = np.full(n, np.nan)
    for i in range(n):
        nb = [j for j in G.neighbors(i) if obs[j]]
        if nb:
            m[i] = weighted_median(y[nb], [G[i][j]["weight"] for j in nb])
    r = y - m
    valid = obs & ~np.isnan(r)
    r0 = np.median(r[valid])
    if s_fixed is None:
        s = max(robust_spread(r[valid]), s_min, 1e-8 * np.std(y))
    else:
        s = s_fixed
    flagged = valid & (np.abs(r - r0) > k * s)
    return flagged, r, s


def detect_outliers_iterative(G, y, mask=None, k=2.5, s_min=0.0, only_first_pass=False):
    """Stage 1, one node at a time: flag the worst node, re-judge the rest without it, repeat.

    s is taken from the first pass and kept fixed.
    Input:  G, y, mask, k, s_min as in detect_outliers;
            only_first_pass  True = only first-pass flags can be confirmed ("prune": removes false
                             flags, never adds new ones).
    Output: (flagged, r, s) as in detect_outliers.
    """
    n = len(y)
    obs = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    first, r, s = detect_outliers(G, y, mask=obs, k=k, s_min=s_min)
    flagged = np.zeros(n, dtype=bool)
    for _ in range(n):
        trusted = obs & ~flagged
        cand, r, _ = detect_outliers(G, y, mask=trusted, k=k, s_fixed=s)
        if only_first_pass:
            cand &= first
        if not cand.any():
            break
        r0 = np.median(r[trusted & ~np.isnan(r)])
        flagged[np.argmax(np.where(cand, np.abs(r - r0), -np.inf))] = True
    return flagged, r, s


def inpaint(L, y, keep, lam=0.0):
    """Stage 2: re-estimate the untrusted nodes from the graph Laplacian.

    Input:
        L     (n, n) graph Laplacian D - W.
        y     float array, length n (values of untrusted nodes are not used when lam = 0).
        keep  bool array, length n, True = trusted.
        lam   0 = keep trusted nodes, solve L_FF x_F = -L_FU y_U for the others;
              > 0 = solve (M + lam * L) x = M y, which also smooths trusted nodes.
    Output: x_hat, float array, length n.
    """
    n = len(y)
    if not keep.any():
        return np.full(n, np.median(y))
    F = ~keep
    if not F.any():
        return y.copy()
    if lam == 0.0:
        x = y.copy()
        x[F] = np.linalg.solve(L[np.ix_(F, F)], -L[np.ix_(F, keep)] @ y[keep])
        return x
    M = np.diag(keep.astype(float))
    return np.linalg.solve(M + lam * L, M @ y)


def flag_nodes(G, y, mask=None, k=2.5, iterative=False, s_floor=0.0):
    """Stage 1 with options.

    Input:  G, y, mask, k as in detect_outliers;
            iterative  False = one pass, True = full iterative, "prune" = re-check first-pass flags only;
            s_floor    relative floor: s >= s_floor * robust_spread(observed y).
    Output: (flagged, r, s) as in detect_outliers.
    """
    obs = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    s_min = s_floor * robust_spread(y[obs])
    if iterative:
        return detect_outliers_iterative(G, y, mask=obs, k=k, s_min=s_min,
                                         only_first_pass=(iterative == "prune"))
    return detect_outliers(G, y, mask=obs, k=k, s_min=s_min)


def solve_two_stage(G, y, mask=None, k=2.5, lam=0.0, iterative="prune", s_floor=0.0):
    """Full method 6 (group contract): flag outliers, then refill them and any hidden nodes.

    Input:  G, y, mask, k as in detect_outliers; lam as in inpaint; iterative, s_floor as in flag_nodes.
            Defaults are the group result (k=2.5, lam=0, iterative="prune"); no tuning needed.
    Output: x_hat, float array, length n.
    """
    y = np.asarray(y, dtype=float)
    obs = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    flagged, _, _ = flag_nodes(G, y, mask=obs, k=k, iterative=iterative, s_floor=s_floor)
    L_G = nx.laplacian_matrix(G, nodelist=range(len(y)), weight="weight").toarray()
    return inpaint(L_G, y, obs & ~flagged, lam=lam)


def flag_outliers(G, y, k=2.5, mask=None, prune=True, return_details=False):
    """Find the suspected outliers.

    Input:
        G               networkx graph, nodes 0..n-1, edge attribute "weight".
        y               float array, length n, observed values.
        k               threshold (default 2.5, keep it); larger = fewer flags.
        mask            bool array, True = observed (default: all observed). Hidden nodes are never used or flagged.
        prune           True = re-check the flags one at a time and drop false ones.
        return_details  True = also return the numbers behind the decision.
    Output:
        flagged         bool array, length n, True = suspected outlier.
        details         (only if return_details) dict with "neighbour_median", "residual", "z"
                        (arrays, length n) and "s" (float), from the first pass.
    """
    y = np.asarray(y, dtype=float)
    obs = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    y_safe = np.where(obs, y, np.nanmedian(y[obs]))
    flagged, _, _ = flag_nodes(G, y_safe, mask=obs, k=k, iterative="prune" if prune else False)
    if not return_details:
        return flagged
    _, r, s = detect_outliers(G, y_safe, mask=obs, k=k)
    z = (r - np.median(r[obs & ~np.isnan(r)])) / s
    return flagged, {"neighbour_median": y_safe - r, "residual": r, "z": z, "s": s}


def refill_outliers(G, y, flagged, lam=0.0):
    """Compute new values for the flagged nodes from their neighbours.

    Input:
        G        networkx graph, same as for flag_outliers.
        y        float array, length n (values of flagged nodes are ignored, may be NaN).
        flagged  bool array, length n, True = do not trust y[i] (outliers and missing nodes).
        lam      0 (default) = only flagged nodes change; > 0 = also smooth the trusted nodes.
    Output:
        x_hat    float array, length n. With lam = 0: x_hat[i] = y[i] for trusted nodes, and the
                 weighted average of the neighbours for flagged nodes.
    """
    y = np.asarray(y, dtype=float)
    flagged = np.asarray(flagged, dtype=bool)
    L_G = nx.laplacian_matrix(G, nodelist=range(len(y)), weight="weight").toarray()
    return inpaint(L_G, np.where(flagged, 0.0, y), ~flagged, lam=lam)
