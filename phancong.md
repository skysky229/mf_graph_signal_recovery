Phân công theo hướng song song: 7 người, 7 phương pháp độc lập
Mỗi người tự giải toàn bộ bài toán bằng một phương pháp riêng. Ai cũng làm đủ 5 bước giống nhau: tự học, viết code, chọn tham số, thử nghiệm, viết thuyết minh. Không ai phải chờ kết quả của người khác. Cuối cùng cả nhóm họp một buổi để so sánh và chọn ra đáp án nộp.
1. “Hợp đồng chung” chốt ngay từ đầu (không ai phải chờ ai)
Mỗi người tự chạy Mục IV với cùng MSSV đại diện để có cùng G và y. Sau đó dán đoạn code chung dưới đây vào file của mình:
import numpy as np, networkx as nx

n = len(y)
L = nx.laplacian_matrix(G, nodelist=range(n), weight="weight").toarray()

# (a) Dữ liệu giả lập: biết x thật, dùng để đo RMSE
def make_synthetic(G, y, seed):
    rng = np.random.default_rng(seed)
    lam, U = np.linalg.eigh(L)
    x = U[:, :5] @ (rng.normal(size=5) / (1 + lam[:5]))
    x = np.median(y) + np.std(y) * x / np.std(x)
    sigma = 0.1 * np.std(y)
    y_syn = x + rng.normal(0, sigma, n)
    idx = rng.choice(n, 3, replace=False)                      # 3 outlier
    y_syn[idx] += rng.choice([-1, 1], 3) * rng.uniform(4, 8, 3) * sigma
    return x, y_syn

def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))

# (b) Chọn tham số trên dữ liệu thật: che 1/5 số đỉnh, dự đoán lại
def cv_score(solve, y, params, k=5, seed=0):
    folds = np.random.default_rng(seed).permutation(n) % k
    err = []
    for f in range(k):
        mask = folds != f                                      # True = được quan sát
        x = solve(G, y, mask=mask, **params)
        err.append(np.abs(x[~mask] - y[~mask]))
    return np.median(np.concatenate(err))                      # median để không bị outlier kéo lệch
Quy ước bắt buộc với mọi người:
Hàm giải có dạng solve_<tên>(G, y, mask=None, **params) -> np.ndarray (dài n). Tham số mask dùng để bỏ qua các đỉnh bị che khi chấm chéo.
Thử nghiệm giả lập dùng seed 0 đến 19, báo cáo RMSE trung bình ± độ lệch chuẩn.
Thuyết minh dài 1–1,5 trang, theo đúng 4 mục của Mục VI (giả thiết → loss → tối ưu và tham số → thực nghiệm và giới hạn).
2. Bảng phân công
#
Phương pháp
Chủ đề tự học
Mô hình
1
Tikhonov / Laplacian regularization
Laplacian, độ trơn xᵀLx, bình phương tối thiểu
min ‖M(x−y)‖² + λxᵀLx, nghiệm đóng (M + λL)x = My
2
Lọc phổ (Graph Fourier)
Trị riêng/vectơ riêng của L, Graph Fourier Transform, tín hiệu band-limited, heat kernel
Giữ K tần số thấp: x = U_K·c, khớp c theo bình phương tối thiểu; hoặc lọc e^{−tL}y
3
Huber + Laplacian
Thống kê bền vững, hàm Huber, thuật toán IRLS
min Σ Huber_δ(xᵢ−yᵢ) + λxᵀLx
4
L1 (LAD) + Laplacian
Chuẩn L1, quy hoạch tuyến tính, thư viện CVXPY
min Σ|xᵢ−yᵢ| + λxᵀLx
5
Graph Total Variation
Total Variation, proximal operator, ADMM
min ‖x−y‖² (hoặc Huber) + λΣwᵢⱼ|xᵢ−xⱼ|
6
Hai giai đoạn: lọc outlier rồi nội suy
Median/MAD, so sánh mỗi đỉnh với láng giềng, nội suy harmonic
(i) Gắn cờ outlier khi |yᵢ − median láng giềng| > k·MAD; (ii) bỏ các đỉnh đó rồi nội suy bằng Laplacian
7
Bayesian: GMRF + nhiễu Student-t
Mô hình xác suất, MAP, thuật toán EM, Gaussian Markov Random Field
Tiên nghiệm x ~ exp(−λxᵀLx), nhiễu Student-t; EM tự ước lượng trọng số từng đỉnh

Các phần nặng nhẹ khác nhau một chút. #1 và #2 có công thức đơn giản hơn, nên mỗi bạn nhận thêm một hình trực quan dùng chung: #1 vẽ y và các đỉnh nghi là outlier trên đồ thị, #2 vẽ phổ của y (hệ số GFT). Hai hình này làm độc lập và chỉ cần dữ liệu gốc.
3. Sản phẩm mỗi người phải nộp (giống nhau cho cả 7)
Một file method_<#>.ipynb chứa hàm solve_..., chạy được từ đầu đến cuối.
Một dòng kết quả gồm: tên phương pháp, tham số tốt nhất, RMSE giả lập (trung bình ± độ lệch chuẩn), cv_score trên dữ liệu thật.
File xhat_<#>.npy: kết quả x_hat trên dữ liệu thật.
Hai hình: đường hội tụ (hoặc đường cong chọn tham số) và hình so sánh y với x_hat trên đồ thị.
Thuyết minh 1–1,5 trang theo khuôn chung.
4. Lịch trình (cả 7 người chạy cùng nhịp)
Thời gian
Mọi người cùng làm
Tuần 1
Tự học chủ đề của mình, viết solve_... và chạy thử trên 1 seed giả lập
Tuần 2
Chạy đủ 20 seed, chọn tham số bằng cv_score, sinh x_hat trên dữ liệu thật
Đầu tuần 3
Viết thuyết minh và vẽ hình
Buổi họp chốt (2–3 giờ)
Mỗi người trình bày 10 phút → ghép bảng so sánh 7 dòng → chọn phương pháp tốt nhất (hoặc lấy trung vị x_hat của 2–3 phương pháp tốt nhất) → cùng ghép notebook, xuất CSV và nộp

5. Ghép notebook cuối cùng
Mục VI: đặt thuyết minh đầy đủ của phương pháp được chọn, kèm bảng so sánh cả 7 phương pháp. Đây là bằng chứng nhóm đã thử nghiệm nghiêm túc.
Mục VII: đặt code của phương pháp được chọn, gồm cả make_synthetic và cv_score.
Thuyết minh của 6 phương pháp còn lại đưa vào phụ lục (file .md hoặc .pdf riêng). Nhờ vậy công sức của ai cũng được ghi nhận như nhau.
Tài liệu chung:
Shuman et al. (2013), IEEE Signal Processing Magazine: nền tảng cho cả nhóm, nhất là #1, #2, #7.
Boyd & Vandenberghe, Convex Optimization, chương 6: cho #3, #4.
Boyd et al. (2011), tài liệu về ADMM: cho #5.
Nếu cần, mình có thể viết sẵn một file khung (template) gồm đoạn code chung và các hàm solve_... để trống cho từng người.

