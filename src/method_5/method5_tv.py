"""Method 5 (Graph Signal Recovery): Graph Total Variation solved with ADMM.

Main function: solve_tv(G, y, mask=None, lam=0.1, loss="l2", ...).
Same code as Section 2.5 of method_5.ipynb (the notebook checks they agree).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def incidence_matrix(G):
    """D (m×n), row e=(i,j): +1 at i, −1 at j (unweighted); w[e] = weight of edge e."""
    n = G.number_of_nodes()
    edges = sorted((min(u, v), max(u, v), d["weight"]) for u, v, d in G.edges(data=True))
    D = np.zeros((len(edges), n)); w = np.empty(len(edges))
    for e, (i, j, wij) in enumerate(edges):
        D[e, i], D[e, j], w[e] = 1.0, -1.0, wij
    return D, w

def soft_threshold(v, tau):
    """prox of tau·|.|, element-wise (tau may be a vector)."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

def huber(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * a - 0.5 * delta**2)

def prox_data(v, y, obs, rho, loss, delta):
    """argmin_z  sum_{i obs} loss(z_i − y_i)/rho + ½||z − v||²  (hidden nodes: z_i = v_i)."""
    z = v.copy()
    r = v[obs] - y[obs]
    if loss == "l2":
        s = rho * r / (1.0 + rho)
    elif loss == "huber":
        s = np.where(np.abs(r) <= delta * (1.0 + 1.0 / rho),
                     rho * r / (1.0 + rho),
                     r - np.sign(r) * delta / rho)
    else:
        raise ValueError(f"unknown loss: {loss}")
    z[obs] = y[obs] + s
    return z

_TV_CACHE = {}
def _tv_operators(G):
    """D, w and the Cholesky factor of I + DᵀD — computed once per graph."""
    hit = _TV_CACHE.get(id(G))
    if hit is None or hit[0] is not G:
        D, w = incidence_matrix(G)
        hit = (G, D, w, cho_factor(np.eye(D.shape[1]) + D.T @ D))
        _TV_CACHE[id(G)] = hit
    return hit[1:]

def tv_objective(x, G, y, mask=None, lam=0.1, loss="l2", delta=0.2):
    D, w, _ = _tv_operators(G)
    obs = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    r = x[obs] - y[obs]
    data = 0.5 * np.sum(r**2) if loss == "l2" else np.sum(huber(r, delta))
    return data + lam * np.sum(w * np.abs(D @ x))

def solve_tv(G, y, mask=None, lam=0.1, loss="l2", delta=0.2, rho=1.0, adaptive=True,
             max_iter=10000, eps_abs=1e-7, eps_rel=1e-5, return_info=False):
    """Graph TV recovery:  min Σ_obs loss(x_i − y_i) + lam·Σ w_ij|x_i − x_j|.

    mask     : bool array of length n, True = observed (None = all observed).
    lam      : model parameter (TV strength) — chosen with cv_score.
    loss     : "l2" (½r²) or "huber" (threshold delta, in the units of y).
    rho      : ADMM algorithm parameter (initial value) — changes speed only, not the solution.
    adaptive : adjust rho to balance the two residuals (Boyd et al. 2011, §3.4.1).
    """
    y = np.asarray(y, dtype=float); n = len(y)
    obs = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    D, w, chol = _tv_operators(G); m = len(w)
    z1 = np.where(obs, y, np.median(y[obs]))       # start hidden nodes at the median
    z2 = D @ z1
    u1, u2 = np.zeros(n), np.zeros(m)
    hist = []
    for k in range(1, max_iter + 1):
        x = cho_solve(chol, (z1 - u1) + D.T @ (z2 - u2))          # x-step
        Dx = D @ x
        z1_old, z2_old = z1, z2
        z1 = prox_data(x + u1, y, obs, rho, loss, delta)           # z1-step (data)
        z2 = soft_threshold(Dx + u2, lam * w / rho)                # z2-step (TV)
        u1 += x - z1; u2 += Dx - z2                                # dual step
        r_pri = np.sqrt(np.sum((x - z1)**2) + np.sum((Dx - z2)**2))
        s_dual = rho * np.linalg.norm((z1 - z1_old) + D.T @ (z2 - z2_old))
        eps_pri = np.sqrt(n + m) * eps_abs + eps_rel * max(np.sqrt(np.sum(x**2) + np.sum(Dx**2)),
                                                             np.sqrt(np.sum(z1**2) + np.sum(z2**2)))
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * rho * np.linalg.norm(u1 + D.T @ u2)
        if return_info:
            hist.append((r_pri, s_dual, eps_pri, eps_dual,
                         tv_objective(x, G, y, obs, lam, loss, delta), rho))
        if r_pri < eps_pri and s_dual < eps_dual:
            break
        # Residual balancing: I + DᵀD has no rho, so changing rho needs no new factorisation;
        # only the scaled dual u = (dual)/rho is rescaled.
        if adaptive and k % 10 == 0:
            if r_pri > 10 * s_dual:
                rho *= 2.0; u1 /= 2.0; u2 /= 2.0
            elif s_dual > 10 * r_pri:
                rho /= 2.0; u1 *= 2.0; u2 *= 2.0
    if return_info:
        return x, {"iters": k, "converged": k < max_iter, "hist": np.array(hist)}
    return x
