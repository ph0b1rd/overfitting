import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# 1. Tải dữ liệu
data = load_breast_cancer()
X = data.data

# Trong bộ dữ liệu gốc: 0 = ác tính, 1 = lành tính.
# Đổi nhãn để lớp dương (1) là khối u ác tính.
y = (data.target == 0).astype(int)

# Chia dữ liệu, giữ tỷ lệ hai lớp tương tự giữa train và test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 2. Tạo pipeline: chuẩn hóa rồi huấn luyện Perceptron
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("perceptron", Perceptron(random_state=42))
])

# 3. Tìm siêu tham số tốt nhất theo F1-score trên cross-validation
param_grid = {
    "perceptron__eta0": [0.001, 0.01, 0.1, 1.0],
    "perceptron__penalty": [None, "l2", "l1", "elasticnet"],
    "perceptron__alpha": [0.0001, 0.001, 0.01],
    "perceptron__max_iter": [1000, 2000]
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)
search.fit(X_train, y_train)

best_model = search.best_estimator_

# 4. Dự đoán trên tập kiểm tra
y_pred = best_model.predict(X_test)

# Tính các độ đo; lớp dương 1 là khối u ác tính
scores = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0)
}

print("Siêu tham số tốt nhất:", search.best_params_)
print(f"F1-score tốt nhất trên cross-validation: {search.best_score_:.4f}")
print("\nKết quả trên tập kiểm tra:")
for name, value in scores.items():
    print(f"{name}: {value:.4f}")

print("\nBáo cáo phân loại:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Lành tính", "Ác tính"],
    zero_division=0
))

# 5. Vẽ biểu đồ kết quả
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Biểu đồ quy trình
axes[0].axis("off")
steps = [
    "1. Tải dữ liệu",
    "2. Chia train / test",
    "3. Chuẩn hóa đặc trưng",
    "4. Tìm tham số bằng GridSearchCV",
    "5. Huấn luyện Perceptron",
    "6. Đánh giá trên tập test"
]
for i, step in enumerate(steps):
    axes[0].text(
        0.05, 0.92 - i * 0.15, step,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f1fa")
    )
axes[0].set_title("Các bước xây dựng mô hình")

# Biểu đồ các độ đo
metric_names = list(scores.keys())
metric_values = list(scores.values())
bars = axes[1].bar(
    metric_names,
    metric_values,
    color=["#4C78A8", "#F58518", "#54A24B", "#E45756"]
)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("Điểm số")
axes[1].set_title("Đánh giá trên tập kiểm tra")
axes[1].grid(axis="y", alpha=0.3)

for bar, value in zip(bars, metric_values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Lành tính", "Ác tính"]
).plot(ax=axes[2], cmap="Blues", colorbar=False)
axes[2].set_title("Ma trận nhầm lẫn")

plt.tight_layout()
plt.show()