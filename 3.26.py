import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Hàm số và đạo hàm
def f(x):
    return x**2 - 4*x + 5

def df(x):
    return 2*x - 4

# Tham số Gradient Descent
x = 5.0
learning_rate = 0.2
so_buoc = 4

# Lưu lại các điểm để in kết quả và vẽ biểu đồ
x_history = [x]
f_history = [f(x)]

print("Đạo hàm: f'(x) = 2x - 4")
print(f"Bước 0: x = {x:.4f}, f(x) = {f(x):.4f}")

# Cập nhật: x_mới = x_cũ - learning_rate * f'(x_cũ)
for buoc in range(1, so_buoc + 1):
    gradient = df(x)
    x = x - learning_rate * gradient

    x_history.append(x)
    f_history.append(f(x))

    print(
        f"Bước {buoc}: gradient = {gradient:.4f}, "
        f"x = {x:.4f}, f(x) = {f(x):.4f}"
    )

# Dùng scikit-learn khớp đường cong từ các điểm dữ liệu mẫu.
# Hai đặc trưng lần lượt là x và x^2.
x_train = np.linspace(0, 5, 100).reshape(-1, 1)
X_features = np.column_stack((x_train[:, 0], x_train[:, 0] ** 2))
y_train = f(x_train[:, 0])

model = LinearRegression()
model.fit(X_features, y_train)

# Tạo đường cong để vẽ
x_plot = np.linspace(0, 5, 200)
X_plot_features = np.column_stack((x_plot, x_plot**2))
y_plot = model.predict(X_plot_features)

# Vẽ đồ thị
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Đồ thị 1: Hàm số và các điểm cập nhật
axes[0].plot(x_plot, y_plot, label="Đường cong khớp bằng scikit-learn")
axes[0].scatter(
    x_history,
    f_history,
    color="red",
    zorder=3,
    label="Các bước Gradient Descent"
)
axes[0].plot(x_history, f_history, "r--", alpha=0.7)

for buoc, (xi, yi) in enumerate(zip(x_history, f_history)):
    axes[0].annotate(
        f"Bước {buoc}\nx={xi:.3f}",
        (xi, yi),
        textcoords="offset points",
        xytext=(5, 8),
        fontsize=8
    )

axes[0].set_title("Gradient Descent trên hàm số f(x)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].grid(True)
axes[0].legend()

# Đồ thị 2: Giá trị hàm số qua từng bước
axes[1].plot(
    range(len(f_history)),
    f_history,
    marker="o",
    color="green"
)
axes[1].set_title("Giá trị hàm số giảm qua các bước")
axes[1].set_xlabel("Bước cập nhật")
axes[1].set_ylabel("f(x)")
axes[1].set_xticks(range(len(f_history)))
axes[1].grid(True)

plt.tight_layout()
plt.show()