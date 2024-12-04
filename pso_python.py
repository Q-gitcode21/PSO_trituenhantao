import numpy as np

# Định nghĩa hàm mục tiêu f(x) = (x - 2)^2 + 3
def ham_muc_tieu(x):
    return (x - 2)*(x - 2) + 3

### INPUT

# Khởi tạo tham số của PSO 
n_particles = 5       # Số lượng hạt
n_iterations = 50   # Số vòng lặp
w = 0.7               # Hệ số quán tính
c1 = 1.4              # Hệ số học hỏi của hạt
c2 = 1.4              # Hệ số học hỏi của nhóm
x_min, x_max = -10, 10 # Khoảng tìm kiếm  Biên dưới -10 , Biên trên 10
#Giới hạn vận tốc  [-1, 1]

# Khởi tạo vị trí và tốc độ ngẫu nhiên
# np.random.uniform (khởi tạo giá trị ngẫu nhiên trong khoảng (low,high,số lượng))
x = np.random.uniform(x_min, x_max, n_particles)  # Vị trí ngẫu nhiên >> tạo ra mảng như: x = [-3.5, 7.2, -8.1, 1.4, 0.9]
v = np.random.uniform(-1, 1, n_particles)         # Tốc độ ngẫu nhiên trong giới hạn vận tốc >> tạo ra mảng như: v = [0.3, -0.5, 0.7, -0.1, -0.8]

# Khởi tạo giá trị tốt nhất cá nhân và toàn cục
p_best = x.copy()  # Mỗi hạt có giá trị tốt nhất ban đầu là vị trí của chính nó . x.copy() = tạo ra bản sao chép của mảng x >> x thay đổi thì p_best ko thay đổi
f_best = ham_muc_tieu(p_best)  # Mảng lưu giá trị hàm mục tiêu tại các vị trí đó
g_best = p_best[np.argmin(f_best)]  # Vị trí tốt nhất toàn cục  .
# np.argmin(f_best) tìm chỉ số của phần tử có giá trị nhỏ nhất trong mảng f_best (tức là hạt nào có giá trị hàm mục tiêu nhỏ nhất).
# p_best[np.argmin(f_best)] lấy giá trị của chỉ số vừa tìm trong p_best >> gán cho g_best VD g_best=3.0 ~ hạt tốt nhất đg là x=3.0
f_gbest = np.min(f_best)  # Giá trị hàm mục tiêu tại vị trí tốt nhất toàn cục ~ lấy giá trị nhỏ nhất trong f_best (giá trị f(x) min)

### Chạy PSO
for iteration in range(n_iterations): # lặp theo số lượng hạt
    # Tính giá trị hàm mục tiêu của từng hạt
    f_x = ham_muc_tieu(x) # f_x là mảng giá trị hàm mục tiêu của các hạt (các x)
    
    # Cập nhật giá trị tốt nhất cá nhân
    better_mask = f_x < f_best  # so sánh f_x (giá trị hiện tại) với  giá trị hàm mục tiêu tại vị trí tốt nhất cá nhân của hạt đó (f_best) >> trả về  mảng T/F
    p_best[better_mask] = x[better_mask]
    # x[better_mask]: Lấy ra các phần tử trong mảng x mà hạt tại vị trí tương ứng có giá trị tốt hơn (better_mask = True). 
    # VD: x[better_mask] = np.array([-2.0]) ~ Chỉ lấy giá trị của hạt thứ 2 (better_mask = True)
    # p_best ban đầu: [3.0, -3.0, 6.0].
    # Sau khi cập nhật  p_best = [3.0, -2.0, 6.0]
    f_best[better_mask] = f_x[better_mask]
    
    # Cập nhật giá trị tốt nhất toàn cục
    g_best = p_best[np.argmin(f_best)]
    f_gbest = np.min(f_best)
    
    # Cập nhật tốc độ và vị trí của các hạt
    r1 = np.random.rand(n_particles) # r1,r2 hệ số ngẫu nhiên 
    r2 = np.random.rand(n_particles)
    # np.random.rand(n_particles) là hàm sử dụng để tạo ra các giá trị ngẫu nhiên trong khoảng [0, 1) ứng với số lượng muốn sinh ngẫu nhiên n_particles 
    # >> mỗi hạt một số ngẫu nhiên trong khoảng [0,1) 
    v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)  # tính vận tốc theo ct 
    x = x + v  # cộng vận tốc vừa tìm được để hạt di chuyển đến vị trí mới 

    # Giới hạn vị trí các hạt trong khoảng [x_min, x_max] ~ (giới hạn giá trị của x trong khoảng tìm kiếm)
    x = np.clip(x, x_min, x_max)
    #Nếu một phần tử trong x nhỏ hơn x_min, nó sẽ được thay thế bằng x_min.
    #Nếu một phần tử trong x lớn hơn x_max, nó sẽ được thay thế bằng x_max.

### OUTPUT
    
    # Hiển thị thông tin của vòng lặp chỉ lấy đại diện các vòng lặp 0,10,20,30,... làm tròn round đến số thập phân thứ 3
    
    if iteration % 10 == 0:
        print(f"Vòng lặp thứ {iteration}: f_gbest = {round(f_gbest, 3)}")
        for i in range(n_particles):
            print(f"Hạt {i + 1}: Vị trí = {round(x[i], 3)}, Vận tốc = {round(v[i], 3)}, f(x) = {round(f_x[i], 3)}")

# Kết quả cuối cùng
print(f"Vị trí tốt nhất x = {round(g_best, 3)}")
print(f"Giá trị tốt nhất f(x) = {round(f_gbest, 3)}")
