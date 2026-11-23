"""Method 3 (Graph Signal Recovery): Huber loss + graph Laplacian, solved by IRLS.

Model (the loss of the group doc):

    min_x  sum_i Huber_delta(x_i - y_i) + lam * x^T L x

    Huber_delta(r) = 0.5 * r^2                    if |r| <= delta
                     delta * |r| - 0.5 * delta^2  if |r| >  delta

`x^T L x = sum_{(i,j)} w_ij (x_i - x_j)^2` is the roughness of x on the graph, so the
second term asks for a smooth signal while the first one asks for a close fit to y.
The Huber loss is quadratic for small residuals (efficient against Gaussian noise) and
linear for large ones (an outlier can only pull with a bounded force, like L1).

Because `d/dx (x^T L x) = 2 L x`, the stationarity condition of one IRLS iteration is

    (W + 2 * lam * L) x = W y,      W = diag(w_1, ..., w_n),

    w_i = 1            if |x_i - y_i| <= delta
        = delta/|r_i|  otherwise.

Main functions: solve_huber, solve_tikhonov, noise_scale.
Defaults (delta = 0.5 * noise_scale, lam = 0.1) are the values used in the group result.
"""
import numpy as np
import networkx as nx

# delta = HUBER_DELTA_ALPHA * noise_scale(G, y): the calibration used in the group result.
HUBER_DELTA_ALPHA = 0.5
DEFAULT_LAM = 0.1


def huber(r, delta):
    """Huber loss element by element.

    Input:  r (array), delta (float > 0 or np.inf).
    Output: array, same shape as r.
    """
    a = np.abs(np.asarray(r, dtype=float))
    if np.isinf(delta):
        return 0.5 * a * a
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def huber_weight(r, delta):
    """IRLS weight w = Huber'(r) / r (so that Huber'(r) = w * r).

    Input:  r (array), delta (float > 0 or np.inf).
    Output: array in (0, 1], same shape as r.
    """
    a = np.abs(np.asarray(r, dtype=float))
    if np.isinf(delta):
        return np.ones_like(a)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-300))


def robust_scale(v):
    """Robust standard deviation: 1.4826 * MAD (median absolute deviation).

    Input:  v (array).
    Output: float.
    """
    v = np.asarray(v, dtype=float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def laplacian(G, n=None):
    """Dense weighted graph Laplacian L = D - W.

    Input:  G (networkx graph, nodes 0..n-1, edge attribute "weight"), n (optional).
    Output: (n, n) float array.
    """
    n = G.number_of_nodes() if n is None else n
    return nx.laplacian_matrix(G, nodelist=range(n), weight="weight").toarray()


def neighbour_mean(G, y, mask=None):
    """Weighted mean of the observed neighbours of every node (NaN if none is observed).

    Input:  G, y (array length n), mask (bool array, True = observed).
    Output: array length n.
    """
    y = np.asarray(y, dtype=float)
    obs = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    m = np.full(len(y), np.nan)
    for i in range(len(y)):
        nb = [j for j in G.neighbors(i) if obs[j]]
        if nb:
            w = np.array([G[i][j]["weight"] for j in nb], dtype=float)
            m[i] = np.sum(w * y[nb]) / np.sum(w)
    return m


def noise_scale(G, y, mask=None):
    """Robust estimate of the node-to-neighbour gap scale.

    r_i = y_i - (weighted mean of the observed neighbours of i),  s0 = 1.4826 * MAD(r).
    On the group data s0 is about the size of the measurement noise (the signal is smooth,
    so neighbours should almost agree).  It is used only to put delta on the data scale.

    Input:  G, y, mask (as in solve_huber).
    Output: float > 0.
    """
    y = np.asarray(y, dtype=float)
    obs = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    r = y - neighbour_mean(G, y, mask=obs)
    valid = obs & ~np.isnan(r)
    s = robust_scale(r[valid])
    # guard against a degenerate value (e.g. a constant y)
    return max(float(s), 1e-8 * float(np.std(y[obs])) + 1e-12)


def huber_objective(G, y, x, delta, lam, mask=None, L=None):
    """Value of the objective F(x) = sum_i Huber_delta(x_i-y_i) + lam * x^T L x.

    Input:  G, y, x, delta, lam, mask; L (optional precomputed Laplacian).
    Output: float.  Nodes with mask False do not enter the data term.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    obs = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    L = laplacian(G, len(y)) if L is None else L
    return float(huber((x - y)[obs], delta).sum() + lam * x @ L @ x)


def solve_huber(G, y, mask=None, delta=None, lam=DEFAULT_LAM, x0=None,
                tol=1e-10, max_iter=1000, return_info=False):
    """Recover the graph signal with Huber loss + Laplacian regularisation, by IRLS.

    Input:
        G      networkx graph, nodes 0..n-1, edge attribute "weight".
        y      float array, length n, observed values.
        mask   bool array, length n, True = observed (default: all observed).
               False = hidden node: its y is ignored and it is re-estimated from the graph
               (used by cv_score).  At least one node must be observed.
        delta  float > 0, the Huber threshold (absolute units of y).
               delta = np.inf gives plain Tikhonov / Laplacian smoothing (method 1).
               None = 0.5 * noise_scale(G, y, mask) (the group result).
        lam    float > 0, weight of the roughness term (the group result: 0.1).
        x0     optional starting point (default y).
        tol    stop when ||x_new - x|| / (1 + ||x_new||) < tol.
        max_iter  iteration cap.
        return_info  True = also return a dict with weights, iterations, objective history,
               KKT residual and delta actually used.

    Output:
        x_hat  float array, length n (finite, one value per node).
        info   dict (only if return_info=True):
               "x", "weights" (w_i, 0 at hidden nodes), "n_iter", "objective" (list),
               "kkt" (max |gradient| at the solution), "delta", "lam".

    Example:
        x_hat = solve_huber(G, y)                       # delta = 0.5*s0, lam = 0.1
        x_hat = solve_huber(G, y, delta=np.inf, lam=0.1)  # Tikhonov reference
        x_hat = solve_huber(G, y, mask=mask, delta=0.337, lam=0.1)  # inside cv_score
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite 1-D array")
    n = len(y)
    obs = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if obs.shape != y.shape:
        raise ValueError("mask must have the same length as y")
    if not obs.any():
        raise ValueError("at least one node must be observed")
    if lam <= 0:
        raise ValueError("lam must be > 0 (with lam = 0 the data term alone is minimised at x = y)")
    if delta is None:
        delta = HUBER_DELTA_ALPHA * noise_scale(G, y, mask=obs)
    delta = float(delta)
    if not (delta > 0):
        raise ValueError("delta must be > 0")

    L = laplacian(G, n)
    x = y.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
    hist = []
    for it in range(1, max_iter + 1):
        w = obs * huber_weight(x - y, delta)
        x_new = np.linalg.solve(np.diag(w) + 2.0 * lam * L, w * y)
        hist.append(huber_objective(G, y, x_new, delta, lam, mask=obs, L=L))
        step = np.linalg.norm(x_new - x) / (1.0 + np.linalg.norm(x_new))
        x = x_new
        if step < tol:
            break

    w = obs * huber_weight(x - y, delta)
    kkt = float(np.max(np.abs(w * (x - y) + 2.0 * lam * L @ x)))
    if not return_info:
        return x
    return x, {"x": x, "weights": w, "n_iter": it, "objective": hist, "kkt": kkt,
               "delta": delta, "lam": float(lam)}


def solve_tikhonov(G, y, mask=None, lam=DEFAULT_LAM):
    """Method 1 reference: min_x sum_i (x_i - y_i)^2 + lam * x^T L x.

    Closed form: (M + 2*lam*L) x = M y, M = diag(mask).  Identical to
    solve_huber(..., delta=np.inf), included so the Huber gain can be measured.

    Input:  G, y, mask, lam (as in solve_huber).
    Output: x_hat, float array of length n.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    obs = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not obs.any():
        raise ValueError("at least one node must be observed")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    L = laplacian(G, n)
    M = obs.astype(float)
    return np.linalg.solve(np.diag(M) + 2.0 * lam * L, M * y)
