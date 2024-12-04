import numpy as np

# Định nghĩa hàm mục tiêu tổng quát cho nhiều tham số (d)
def ham_muc_tieu(x):
   # Hàm mục tiêu: f(x, y, z) = (x^2 + y^2 + z^2) * sin(x) + cos(y) * z + 2
    return (x[0]**2 + x[1]**2 + x[2]**2) * np.sin(x[0]) + np.cos(x[1]) * x[2] + 2

### INPUT

# Khởi tạo tham số của PSO
n_particles = 20      # Số lượng hạt
n_iterations = 100     # Số vòng lặp
w = 0.7               # Hệ số quán tính
c1 = 1.4              # Hệ số học hỏi của hạt
c2 = 1.4              # Hệ số học hỏi của nhóm
d = 3                 # Số chiều (số tham số cần tối ưu, ví dụ: x, y, z)
x_min, x_max = -10, 10 # Khoảng tìm kiếm cho mỗi tham số (biên dưới, biên trên)

# Giới hạn vận tốc
v_min, v_max = -2, 2  # Giới hạn vận tốc cho từng tham số

# Khởi tạo vị trí và tốc độ ngẫu nhiên của các hạt
x = np.random.uniform(x_min, x_max, (n_particles, d))  # Vị trí ngẫu nhiên của các hạt, mảng (n_particles, d)
v = np.random.uniform(v_min, v_max, (n_particles, d))  # Vận tốc ngẫu nhiên của các hạt, mảng (n_particles, d)

# Khởi tạo giá trị tốt nhất cá nhân và toàn cục
p_best = x.copy()  # Vị trí tốt nhất cá nhân ban đầu là vị trí của chính nó
f_best = np.apply_along_axis(ham_muc_tieu, 1, p_best)  # Giá trị hàm mục tiêu tại vị trí đó
g_best = p_best[np.argmin(f_best)]  # Vị trí tốt nhất toàn cục
f_gbest = np.min(f_best)  # Giá trị hàm mục tiêu tại vị trí tốt nhất toàn cục

# Thực hiện PSO trong n_iterations vòng lặp
for iteration in range(n_iterations):
    # Tính giá trị hàm mục tiêu tại vị trí hiện tại của các hạt
    f_x = np.apply_along_axis(ham_muc_tieu, 1, x)

    # Cập nhật giá trị tốt nhất cá nhân
    better_mask = f_x < f_best
    p_best[better_mask] = x[better_mask]
    f_best[better_mask] = f_x[better_mask]

    # Cập nhật giá trị tốt nhất toàn cục
    g_best = p_best[np.argmin(f_best)]
    f_gbest = np.min(f_best)

    # Cập nhật vận tốc và vị trí của các hạt
    r1 = np.random.rand(n_particles, d)  # Hệ số ngẫu nhiên 1
    r2 = np.random.rand(n_particles, d)  # Hệ số ngẫu nhiên 2

    v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)  # Cập nhật vận tốc

    # Giới hạn vận tốc trong khoảng [-v_max, v_max]
    v = np.clip(v, v_min, v_max)

    # Cập nhật vị trí
    x = x + v

    # Giới hạn vị trí trong khoảng [x_min, x_max]
    x = np.clip(x, x_min, x_max)

    # In thông tin mỗi 10 vòng lặp
    if iteration % 10 == 0:
        print(f"Vòng lặp thứ {iteration}: f_gbest = {round(f_gbest, 3)}")
        print(f"Vị trí tốt nhất g_best = {np.round(g_best, 3)}")

# Kết quả cuối cùng
print(f"Vị trí tốt nhất cuối cùng: {np.round(g_best, 3)}")
print(f"Giá trị hàm mục tiêu nhỏ nhất f(x): {round(f_gbest, 3)}")
