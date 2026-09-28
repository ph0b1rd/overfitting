import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=1.0, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều.")
        if len(X) != len(y):
            raise ValueError("Số mẫu trong X và y phải bằng nhau.")
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Nhãn y chỉ được chứa -1 hoặc +1.")

        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.errors_per_epoch = []

        for _ in range(self.n_epochs):
            errors = 0

            for xi, yi in zip(X, y):
                score = np.dot(xi, self.weights) + self.bias
                prediction = 1 if score >= 0 else -1

                if prediction != yi:
                    self.weights += self.learning_rate * yi * xi
                    self.bias += self.learning_rate * yi
                    errors += 1

            self.errors_per_epoch.append(errors)

            if errors == 0:
                break

        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi fit trước.")

        X = np.asarray(X, dtype=float)
        scores = np.dot(X, self.weights) + self.bias
        return np.where(scores >= 0, 1, -1)

    def plot_decision_boundary(self, X, y):
        """Vẽ dữ liệu và đường biên; chỉ áp dụng cho dữ liệu 2 chiều."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[1] != 2:
            raise ValueError("Chỉ vẽ được đường biên khi X có đúng 2 đặc trưng.")
        if self.weights is None:
            raise ValueError("Hãy huấn luyện mô hình bằng fit trước.")

        # Tạo lưới để tô màu hai vùng dự đoán
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.predict(grid).reshape(xx.shape)

        plt.contourf(
            xx, yy, predictions,
            levels=[-1.5, 0, 1.5],
            colors=["#f7c6c7", "#c9e7d2"],
            alpha=0.6
        )

        # Vẽ các điểm thuộc từng lớp
        for label, color, name in [
            (-1, "red", "Nhãn -1"),
            (1, "blue", "Nhãn +1")
        ]:
            mask = y == label
            plt.scatter(
                X[mask, 0], X[mask, 1],
                color=color,
                edgecolors="black",
                s=100,
                label=name
            )

        # Đường biên: w1*x1 + w2*x2 + bias = 0
        w1, w2 = self.weights

        if abs(w2) > 1e-12:
            x_line = np.linspace(x_min, x_max, 300)
            y_line = -(w1 * x_line + self.bias) / w2
            plt.plot(x_line, y_line, "k--", label="Đường biên phân lớp")
        elif abs(w1) > 1e-12:
            x_vertical = -self.bias / w1
            plt.axvline(x_vertical, color="black", linestyle="--",
                        label="Đường biên phân lớp")

        plt.xlabel("Đặc trưng 1")
        plt.ylabel("Đặc trưng 2")
        plt.title("Perceptron: vùng dự đoán và đường phân lớp")
        plt.grid(True)
        plt.legend()
        plt.show()


# Dữ liệu AND: nhãn -1 hoặc +1
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_train = np.array([-1, -1, -1, 1])

model = Perceptron(learning_rate=1.0, n_epochs=20)
model.fit(X_train, y_train)

print("Trọng số:", model.weights)
print("Bias:", model.bias)
print("Nhãn dự đoán trên dữ liệu huấn luyện:", model.predict(X_train))

# Vẽ đường biên phân lớp
model.plot_decision_boundary(X_train, y_train)

# Vẽ số lỗi qua từng epoch
plt.plot(range(1, len(model.errors_per_epoch) + 1),
         model.errors_per_epoch, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Số mẫu phân lớp sai")
plt.title("Quá trình huấn luyện Perceptron")
plt.grid(True)
plt.show()