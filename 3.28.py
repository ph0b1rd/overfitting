import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Dữ liệu: x đã có phần tử bias ở cuối
w = np.array([-2.0, 1.0, 0.0])
x = np.array([2.0, 3.0, 1.0])
y_thuc_te = 1
learning_rate = 1.0

# Tạo Perceptron với hai đặc trưng (x1, x2).
# Bias được biểu diễn bằng intercept_ của scikit-learn.
model = Perceptron(fit_intercept=True)
model.classes_ = np.array([-1, 1])
model.coef_ = w[:2].reshape(1, -1)
model.intercept_ = np.array([w[2]])

# 1. Kiểm tra phân lớp ban đầu
gia_tri_ban_dau = np.dot(w, x)
y_du_doan_ban_dau = model.predict(x[:2].reshape(1, -1))[0]
bi_phan_lop_sai = y_du_doan_ban_dau != y_thuc_te

print(f"w^T x ban đầu = {gia_tri_ban_dau:.1f}")
print(f"Nhãn dự đoán ban đầu = {y_du_doan_ban_dau}")
print(f"Nhãn thực tế = {y_thuc_te}")
print(f"Mẫu bị phân lớp sai: {bi_phan_lop_sai}")

# 2. Nếu phân lớp sai, cập nhật: w_mới = w + η * y * x
if bi_phan_lop_sai:
    w = w + learning_rate * y_thuc_te * x

# Cập nhật trọng số của mô hình scikit-learn để kiểm tra lại
model.coef_ = w[:2].reshape(1, -1)
model.intercept_ = np.array([w[2]])

# 3. Tính lại tích vô hướng và dự đoán
gia_tri_sau_cap_nhat = np.dot(w, x)
y_du_doan_sau_cap_nhat = model.predict(x[:2].reshape(1, -1))[0]

print(f"\nTrọng số sau cập nhật = {w}")
print(f"w^T x sau cập nhật = {gia_tri_sau_cap_nhat:.1f}")
print(f"Nhãn dự đoán sau cập nhật = {y_du_doan_sau_cap_nhat}")

# Vẽ đường biên: w1*x1 + w2*x2 + bias = 0
x1 = np.linspace(-1, 4, 300)

def duong_bien(weights):
    w1, w2, bias = weights
    if abs(w2) < 1e-12:
        return None  # Đường biên đứng, xử lý riêng bên dưới
    return -(w1 * x1 + bias) / w2

plt.figure(figsize=(8, 6))

# Đường biên trước cập nhật: -2*x1 + x2 = 0
x2_truoc = duong_bien(np.array([-2.0, 1.0, 0.0]))
plt.plot(x1, x2_truoc, "--", label="Đường biên trước cập nhật")

# Đường biên sau cập nhật: 4*x2 + 1 = 0
x2_sau = duong_bien(w)
plt.plot(x1, x2_sau, label="Đường biên sau cập nhật")

# Điểm dữ liệu (2, 3)
plt.scatter(
    x[0], x[1],
    color="red",
    marker="o",
    s=100,
    label=f"Điểm (2, 3), nhãn thực tế +1"
)
plt.annotate(
    f"Trước: dự đoán {y_du_doan_ban_dau}\nSau: dự đoán {y_du_doan_sau_cap_nhat}",
    (x[0], x[1]),
    xytext=(10, 10),
    textcoords="offset points"
)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Cập nhật Perceptron cho một mẫu dữ liệu")
plt.xlim(-1, 4)
plt.ylim(-1, 5)
plt.grid(True)
plt.legend()
plt.show()