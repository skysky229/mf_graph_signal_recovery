"""Method 4 (Graph Signal Recovery): LAD (L1) data fit + Laplacian regularization.

    x_hat = argmin_x  sum_{i observed} |x_i - y_i| + lam * x^T L x

Main functions: solve_lad (CVXPY + Clarabel, exact), solve_lad_irls (IRLS / majorize-minimize, cross-check),
solve_tikhonov (L2 baseline), lambda0 (below this lam the LAD solution is y itself), kkt_violation.
Same code as Section 3 of src/method_4/method_4.ipynb.

Example
-------
    from method4_lad import solve_lad, lambda0
    x_hat = solve_lad(G, y, lam=0.2234)                    # group result on case 20260001 (= 3.16 * lambda0)
    x_hat = solve_lad(G, y, mask=mask, lam=0.2234)         # inside cv_score
"""
import numpy as np
import networkx as nx
import cvxpy as cp


# ----------------------------------------------------------------------------
# Tiện ích đồ thị
# ----------------------------------------------------------------------------
def laplacian(G):
    n = G.number_of_nodes()
    return nx.laplacian_matrix(G, nodelist=range(n), weight="weight").toarray()


def weighted_incidence(G):
    """D sao cho x^T L x = ||D x||^2  (D = sqrt(W) B, B là ma trận liên thuộc cạnh-đỉnh)."""
    n = G.number_of_nodes()
    edges = list(G.edges(data="weight"))
    D = np.zeros((len(edges), n))
    for k, (u, v, w) in enumerate(edges):
        s = np.sqrt(w)
        D[k, u], D[k, v] = s, -s
    return D


def _mask(n, mask):
    return np.ones(n, bool) if mask is None else np.asarray(mask, bool)


def lad_objective(x, y, L, lam, mask=None):
    m = _mask(len(y), mask)
    return float(np.abs(x - y)[m].sum() + lam * x @ L @ x)


def lambda0(G, y):
    """Với lam < lambda0, x = y thỏa KKT  =>  nghiệm LAD chính là y (không khử nhiễu)."""
    return 1.0 / (2.0 * np.abs(laplacian(G) @ y).max())


# ----------------------------------------------------------------------------
# Solver 1: CVXPY (DPP – biên dịch 1 lần cho mỗi đồ thị, sau đó chỉ đổi tham số)
# ----------------------------------------------------------------------------
_CACHE = {}


def _get_problem(G):
    # key theo nội dung đồ thị (không dùng id(G): id có thể bị tái sử dụng)
    key = (G.number_of_nodes(), tuple(sorted((min(u, v), max(u, v), round(w, 12))
                                         for u, v, w in G.edges(data="weight"))))
    if key not in _CACHE:
        n = G.number_of_nodes()
        D = weighted_incidence(G)
        x = cp.Variable(n)
        my_p = cp.Parameter(n)                 # m * y
        m_p = cp.Parameter(n, nonneg=True)     # 1 = quan sát, 0 = bị che
        lam_p = cp.Parameter(nonneg=True)
        # viết m*x - m*y (thay vì m*(x-y)) để bài toán là DPP -> chỉ biên dịch 1 lần
        obj = cp.norm1(cp.multiply(m_p, x) - my_p) + lam_p * cp.sum_squares(D @ x)
        _CACHE[key] = (cp.Problem(cp.Minimize(obj)), x, my_p, m_p, lam_p)
    return _CACHE[key]


def solve_lad(G, y, mask=None, lam=1.0, solver="CLARABEL"):
    prob, x, my_p, m_p, lam_p = _get_problem(G)
    m = _mask(len(y), mask).astype(float)
    m_p.value = m
    my_p.value = m * np.asarray(y, float)
    lam_p.value = float(lam)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY status = {prob.status}")
    return np.asarray(x.value, float).ravel()


# ----------------------------------------------------------------------------
# Solver 2: IRLS (majorize-minimize)
#   |r| <= r^2 / (2|r_k|) + |r_k|/2   =>   mỗi bước giải  (W + 2 lam L) x = W y,
#   W = diag(m_i / max(|r_i|, eps)).  eps giảm dần để tiến tới bài toán L1 gốc.
# ----------------------------------------------------------------------------
def solve_lad_irls(G, y, mask=None, lam=1.0, eps0=1e-1, eps_min=1e-10,
                   max_iter=3000, tol_stage=1e-7, tol=1e-12, return_history=False):
    y = np.asarray(y, float)
    n = len(y)
    m = _mask(n, mask).astype(float)
    L = laplacian(G)
    # khởi tạo bằng Tikhonov (L2)
    x = np.linalg.solve(np.diag(m) + lam * L, m * y)
    eps = eps0
    hist = [lad_objective(x, y, L, lam, mask)]
    for _ in range(max_iter):
        w = m / np.maximum(np.abs(x - y), eps)
        x = np.linalg.solve(np.diag(w) + 2 * lam * L, w * y)
        f = lad_objective(x, y, L, lam, mask)
        hist.append(f)
        rel = abs(hist[-2] - f) / max(1.0, f)
        if eps <= eps_min and rel <= tol:
            break                                   # dừng: eps nhỏ nhất và f không đổi
        if eps > eps_min and rel <= tol_stage:
            eps = max(eps * 0.1, eps_min)           # giai đoạn hiện tại đã hội tụ -> giảm eps
    return (x, np.array(hist)) if return_history else x


# ----------------------------------------------------------------------------
# Baseline L2: min ||M(x-y)||^2 + lam x^T L x  ->  (M + lam L) x = M y
# ----------------------------------------------------------------------------
def solve_tikhonov(G, y, mask=None, lam=1.0):
    m = _mask(len(y), mask).astype(float)
    L = laplacian(G)
    return np.linalg.solve(np.diag(m) + lam * L, m * np.asarray(y, float))


# ----------------------------------------------------------------------------
# Kiểm tra KKT:  0 ∈ s + 2 lam L x,  s_i = sign(x_i - y_i) nếu x_i != y_i, s_i ∈ [-1,1] nếu bằng
# Trả về max vi phạm (≈0 nếu tối ưu) và số đỉnh "dính" x_i == y_i.
# ----------------------------------------------------------------------------
def kkt_violation(G, y, x, lam, mask=None, tol=1e-5):
    m = _mask(len(y), mask)
    g = 2 * lam * laplacian(G) @ x          # cần s = -g
    r = x - y
    viol = np.zeros(len(y))
    tight = m & (np.abs(r) <= tol)
    loose = m & ~tight
    viol[~m] = np.abs(g[~m])                                   # đỉnh bị che: s=0
    viol[tight] = np.maximum(np.abs(g[tight]) - 1, 0)          # |g| <= 1
    viol[loose] = np.abs(g[loose] + np.sign(r[loose]))         # g = -sign(r)
    return float(viol.max()), int(tight.sum())
