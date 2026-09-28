import numpy as np
from sklearn.linear_model import Perceptron

# Trọng số: [w1, w2, bias]
w = np.array([1, 2, -10])

# Điểm dữ liệu đã có bias: [x1, x2, 1]
x = np.array([3, 4, 1])

# Nhãn thực tế
y_thuc_te = -1

# 1. Tính w^T x
tich_vo_huong = np.dot(w, x)
print(f"1. w^T x = {tich_vo_huong}")

# 2. Dự đoán theo quy tắc Perceptron
# Quy ước: w^T x >= 0 => +1; w^T x < 0 => -1
y_du_doan = 1 if tich_vo_huong >= 0 else -1
print(f"2. Nhãn dự đoán = {y_du_doan}")

# 3. Kiểm tra phân lớp sai
phan_lop_sai = y_du_doan != y_thuc_te
print(f"3. Điểm dữ liệu bị phân lớp sai: {phan_lop_sai}")

# Minh họa Perceptron của scikit-learn
# fit_intercept=False vì bias đã được đưa vào x
model = Perceptron(fit_intercept=False)

# Gán trọng số có sẵn; scikit-learn yêu cầu mô hình được fit trước
model.classes_ = np.array([-1, 1])
model.coef_ = w[:2].reshape(1, -1)
model.intercept_ = np.array([w[2]])

# sklearn nhận các đặc trưng chưa thêm bias,
# vì vậy chỉ truyền [3, 4] vào predict()
x_sklearn = x[:2].reshape(1, -1)
y_sklearn = model.predict(x_sklearn)[0]

print(f"Nhãn dự đoán bằng scikit-learn = {y_sklearn}")