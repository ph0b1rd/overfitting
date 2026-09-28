"""So sanh underfitting, good fit va overfitting tren du lieu gia nha Ha Noi."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from overfitting_demo import RANDOM_STATE, evaluate, load_data, make_model


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def add_labels(axis) -> None:
    """Them nhan gia tri vao cac cot trong mot truc."""
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f", padding=3, fontsize=9)


def plot_fit_comparison(result_table: pd.DataFrame) -> None:
    """Ve dashboard so sanh ba muc do phu hop cua mo hinh."""
    labels = ["Underfit\n(bac 1)", "Good fit\n(Ridge bac 2)", "Overfit\n(bac 8)"]
    colors = ["#f4a261", "#2a9d8f", "#e76f51"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle(
        "Nhan dien Underfit, Good fit va Overfit",
        fontsize=18,
        fontweight="bold",
    )

    result_table.plot(
        x="Loai_mo_hinh",
        y=["R2_train", "R2_test"],
        kind="bar",
        ax=axes[0, 0],
        color=["#457b9d", "#2a9d8f"],
        width=0.7,
    )
    axes[0, 0].set_title("R² train va test")
    axes[0, 0].set_ylabel("R²")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_xticklabels(labels, rotation=0)
    axes[0, 0].legend(["Train", "Test"], frameon=False)
    add_labels(axes[0, 0])

    result_table.plot(
        x="Loai_mo_hinh",
        y="Khoang_cach_R2",
        kind="bar",
        ax=axes[0, 1],
        color=colors,
        width=0.7,
        legend=False,
    )
    axes[0, 1].axhline(0, color="#264653", linewidth=1)
    axes[0, 1].set_title("Khoang cach train - test")
    axes[0, 1].set_ylabel("Do lech R²")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_xticklabels(labels, rotation=0)
    add_labels(axes[0, 1])

    result_table.plot(
        x="Loai_mo_hinh",
        y=["RMSE_test", "MAE_test"],
        kind="bar",
        ax=axes[1, 0],
        color=["#264653", "#a8dadc"],
        width=0.7,
    )
    axes[1, 0].set_title("Sai so tren tap test")
    axes[1, 0].set_ylabel("Gia tri sai so")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_xticklabels(labels, rotation=0)
    axes[1, 0].legend(["RMSE", "MAE"], frameon=False)
    add_labels(axes[1, 0])

    positions = range(len(result_table))
    axes[1, 1].bar(positions, result_table["RMSE_test"], color=colors)
    axes[1, 1].set_title("RMSE test: thap hon la tot hon")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_xticks(list(positions), labels, rotation=0)
    for bar in axes[1, 1].patches:
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            fontsize=9,
        )

    for axis in axes.ravel():
        axis.margins(y=0.15)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(OUTPUT_DIR / "fit_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    models = {
        "Underfit": make_model(LinearRegression(), degree=1),
        "Good fit": make_model(Ridge(alpha=0.1), degree=2),
        "Overfit": make_model(LinearRegression(), degree=8),
    }
    results = [
        evaluate(name, model, X_train, X_test, y_train, y_test)
        for name, model in models.items()
    ]
    result_table = pd.DataFrame(results).rename(columns={"Mo_hinh": "Loai_mo_hinh"})

    OUTPUT_DIR.mkdir(exist_ok=True)
    result_table.to_csv(OUTPUT_DIR / "fit_comparison.csv", index=False)
    print("=== SO SANH UNDERFIT, GOOD FIT VA OVERFIT ===")
    print(result_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Da luu ket qua tai: {OUTPUT_DIR / 'fit_comparison.csv'}")
    print(f"Da luu bieu do tai: {OUTPUT_DIR / 'fit_comparison.png'}")

    plot_fit_comparison(result_table)


if __name__ == "__main__":
    main()
