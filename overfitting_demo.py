"""Minh hoa overfitting va cac cach khac phuc tren du lieu gia nha Ha Noi."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Gia_Nha_Ha_Noi.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(DATA_PATH)
    target = "Gia_Nha"
    if target not in data.columns:
        raise ValueError(f"Khong tim thay cot muc tieu {target!r}.")
    data = data.drop(columns=["ID"], errors="ignore").dropna()
    return data.drop(columns=[target]), data[target]


def make_model(regressor, degree: int) -> Pipeline:
    numeric_features = [
        "Dien_Tich",
        "So_Phong_Ngu",
        "Khoang_Cach_Trung_Tam",
    ]
    categorical_features = ["Quan"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        sparse_threshold=0,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def evaluate(name: str, model: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    return {
        "Mo_hinh": name,
        "R2_train": r2_score(y_train, train_prediction),
        "R2_test": r2_score(y_test, test_prediction),
        "Khoang_cach_R2": r2_score(y_train, train_prediction) - r2_score(y_test, test_prediction),
        "MAE_test": mean_absolute_error(y_test, test_prediction),
        "RMSE_test": mean_squared_error(y_test, test_prediction) ** 0.5,
    }


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    models = {
        "Overfit: da thuc bac 8": make_model(LinearRegression(), degree=8),
        "Giam do phuc tap: da thuc bac 2": make_model(LinearRegression(), degree=2),
        "Ridge: bac 8, alpha=10": make_model(Ridge(alpha=10.0), degree=8),
        "Lasso: bac 8, alpha=0.01": make_model(Lasso(alpha=0.01, max_iter=100000), degree=8),
    }

    results = [
        evaluate(name, model, X_train, X_test, y_train, y_test)
        for name, model in models.items()
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=make_model(Ridge(), degree=2),
        param_grid={
            "preprocessor__num__polynomial__degree": [1, 2, 3, 4],
            "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned_result = evaluate("Tu dong chon bac + Ridge", search.best_estimator_, X_train, X_test, y_train, y_test)
    results.append(tuned_result)

    result_table = pd.DataFrame(results).sort_values("RMSE_test")
    OUTPUT_DIR.mkdir(exist_ok=True)
    result_table.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    print("=== KET QUA SO SANH ===")
    print(result_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== THAM SO TOT NHAT TU CROSS-VALIDATION ===")
    print(search.best_params_)
    print(f"CV RMSE tot nhat: {-search.best_score_:.4f}")
    print(f"Da luu ket qua tai: {OUTPUT_DIR / 'model_results.csv'}")

    plot_results(result_table)


def plot_results(result_table: pd.DataFrame) -> None:
    labels = [
        textwrap.fill(name.replace(": ", ":\n"), width=18)
        for name in result_table["Mo_hinh"]
    ]
    colors = ["#2a9d8f" if value == result_table["RMSE_test"].min() else "#8ecae6"
              for value in result_table["RMSE_test"]]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    figure.suptitle("So sanh mo hinh va nhan dien overfitting", fontsize=18, fontweight="bold")

    metric_axes = axes.ravel()
    result_table.plot(
        x="Mo_hinh", y=["R2_train", "R2_test"], kind="bar", ax=metric_axes[0],
        color=["#e76f51", "#2a9d8f"], width=0.72,
    )
    metric_axes[0].set_title("R² train va test")
    metric_axes[0].set_ylabel("R²")
    metric_axes[0].set_xlabel("")
    metric_axes[0].tick_params(axis="x", labelrotation=0)
    metric_axes[0].set_xticklabels(labels)
    metric_axes[0].legend(["Train", "Test"], frameon=False)

    result_table.plot(
        x="Mo_hinh", y="Khoang_cach_R2", kind="bar", ax=metric_axes[1],
        color="#f4a261", width=0.72, legend=False,
    )
    metric_axes[1].axhline(0, color="#264653", linewidth=1)
    metric_axes[1].set_title("Khoang cach R² train - test")
    metric_axes[1].set_ylabel("Do lech R²")
    metric_axes[1].set_xlabel("")
    metric_axes[1].tick_params(axis="x", labelrotation=0)
    metric_axes[1].set_xticklabels(labels)

    result_table.plot(
        x="Mo_hinh", y=["RMSE_test", "MAE_test"], kind="bar", ax=metric_axes[2],
        color=["#457b9d", "#a8dadc"], width=0.72,
    )
    metric_axes[2].set_title("Sai so tren tap test")
    metric_axes[2].set_ylabel("Gia tri sai so")
    metric_axes[2].set_xlabel("")
    metric_axes[2].tick_params(axis="x", labelrotation=0)
    metric_axes[2].set_xticklabels(labels)
    metric_axes[2].legend(["RMSE", "MAE"], frameon=False)

    sorted_results = result_table.sort_values("RMSE_test", ascending=True)
    metric_axes[3].barh(
        [textwrap.fill(name, width=22) for name in sorted_results["Mo_hinh"]],
        sorted_results["RMSE_test"], color=[colors[result_table.index.get_loc(index)] for index in sorted_results.index],
    )
    metric_axes[3].set_title("Xep hang theo RMSE test")
    metric_axes[3].set_xlabel("RMSE (thap hon la tot hon)")
    metric_axes[3].invert_yaxis()

    for axis in metric_axes[:3]:
        for container in axis.containers:
            axis.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
        axis.margins(y=0.15)
    for bar in metric_axes[3].patches:
        metric_axes[3].text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}",
            va="center",
            fontsize=9,
        )

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT_DIR / "overfitting_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
