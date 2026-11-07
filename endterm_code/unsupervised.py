"""Train-only KMeans and PCA for interpretable numeric vehicle descriptions.

Asking price is read only after clustering for descriptive profiling. Neither
price nor categorical columns enter the imputer, scaler, KMeans, or PCA.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["year", "log_km", "engine_liters_clean"]
COLORS = ["#16734B", "#E09C36", "#4577A5", "#A8648F", "#718D39", "#B46A4D", "#5B568D", "#527B7A"]


def _records(frame):
    """Convert missing observed medians to JSON null rather than nonstandard NaN."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


def run_unsupervised(train_frame: pd.DataFrame, output_dir: Path, seed=42):
    """Fit KMeans/PCA on the supplied training listings and export diagnostics.

    Select k with the highest training silhouette among 2..8; break an exact
    tie in favour of the smaller k. This is one geometric heuristic, not proof
    of natural market segments or a universally optimal segmentation.
    """
    required = ["ad_id", "price", "mileage_km", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in train_frame]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")
    if len(train_frame) < 9:
        raise ValueError("At least nine training listings are needed to compare k=2..8.")
    if train_frame["ad_id"].isna().any() or train_frame["ad_id"].duplicated().any():
        raise ValueError("Training ad_id values must be nonmissing and unique.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = train_frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.isna().all().any():
        empty = numeric.columns[numeric.isna().all()].tolist()
        raise ValueError(f"A training median is undefined for entirely missing features: {empty}")

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)
    if len(np.unique(scaled, axis=0)) < 8:
        raise ValueError("At least eight distinct numeric descriptions are needed for k=2..8.")

    models = {}
    selection_rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = model.fit_predict(scaled)
        models[k] = model
        selection_rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(scaled, labels)),
        })
    best = max(selection_rows, key=lambda row: (row["silhouette"], -row["k"]))
    selected_k = best["k"]
    model = models[selected_k]
    labels = model.labels_
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output_dir / "cluster_selection.csv", index=False)

    # The same training-only numeric representation is used for PCA.
    pca = PCA(n_components=3, svd_solver="full")
    coordinates = pca.fit_transform(scaled)
    variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance)
    assignments = pd.DataFrame({
        "ad_id": train_frame["ad_id"].to_numpy(),
        "cluster": labels,
        "PC1": coordinates[:, 0],
        "PC2": coordinates[:, 1],
        "PC3": coordinates[:, 2],
    })
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
    assignments.to_csv(output_dir / "pca_projection.csv", index=False)

    scaled_centroids = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLUMNS)
    scaled_centroids.insert(0, "cluster", range(selected_k))
    scaled_centroids.to_csv(output_dir / "cluster_centroids_scaled.csv", index=False)
    original_centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_), columns=FEATURE_COLUMNS
    )
    original_centroids.insert(0, "cluster", range(selected_k))
    original_centroids["mileage_km_from_log_centroid"] = np.expm1(original_centroids["log_km"])
    original_centroids.to_csv(output_dir / "cluster_centroids_original.csv", index=False)

    # Original observed values, not imputed values, describe each fitted cluster.
    profile_values = numeric[["year", "engine_liters_clean"]].copy()
    for column in ["mileage_km", "price"]:
        profile_values[column] = pd.to_numeric(train_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    profile_values["cluster"] = labels
    profiles = profile_values.groupby("cluster").agg(
        count=("year", "size"),
        median_year=("year", "median"),
        median_mileage_km=("mileage_km", "median"),
        median_engine_liters=("engine_liters_clean", "median"),
        median_price_kzt=("price", "median"),
        observed_year_count=("year", "count"),
        observed_mileage_count=("mileage_km", "count"),
        observed_engine_count=("engine_liters_clean", "count"),
        observed_price_count=("price", "count"),
    ).reset_index()
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)

    component_names = ["PC1", "PC2", "PC3"]
    loadings = pd.DataFrame(pca.components_, columns=FEATURE_COLUMNS)
    loadings.insert(0, "component", component_names)
    loadings.to_csv(output_dir / "pca_loadings.csv", index=False)
    variance_frame = pd.DataFrame({
        "component": component_names,
        "explained_variance_ratio": variance,
        "cumulative_explained_variance_ratio": cumulative_variance,
        "explained_variance": pca.explained_variance_,
    })
    variance_frame.to_csv(output_dir / "pca_variance.csv", index=False)

    with plt.rc_context({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].plot(selection["k"], selection["inertia"], marker="o", color=COLORS[0])
        axes[0].set(title="Within-cluster squared distances", xlabel="Number of clusters k", ylabel="Inertia (scaled space)")
        axes[1].plot(selection["k"], selection["silhouette"], marker="o", color=COLORS[0])
        axes[1].scatter([selected_k], [best["silhouette"]], s=95, color=COLORS[1], zorder=5,
                        label=f"Selected k={selected_k}")
        axes[1].set(title="Training silhouette", xlabel="Number of clusters k", ylabel="Silhouette score")
        axes[1].legend(frameon=False)
        for axis in axes:
            axis.set_xticks(range(2, 9))
            axis.grid(axis="y", alpha=0.16)
        fig.suptitle("KMeans on year, log mileage and engine size", y=1.01)
        fig.tight_layout()
        fig.savefig(output_dir / "cluster_selection.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7.8, 4.4))
        for cluster in range(selected_k):
            mask = labels == cluster
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=17, alpha=0.55,
                         color=COLORS[cluster], linewidths=0,
                         label=f"Cluster {cluster} (n={int(mask.sum())})")
        axis.set(
            title=f"Numeric vehicle descriptions: PCA projection (k={selected_k})",
            xlabel=f"PC1 ({variance[0]:.1%} variance)",
            ylabel=f"PC2 ({variance[1]:.1%} variance)",
        )
        axis.legend(frameon=False, markerscale=1.4, fontsize=9)
        axis.grid(alpha=0.14)
        fig.tight_layout()
        fig.savefig(output_dir / "pca_scatter.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(6.6, 3.8))
        axis.bar(component_names, variance, color=COLORS[0], alpha=0.85, label="Individual variance")
        axis.plot(component_names, cumulative_variance, marker="o", color=COLORS[1], label="Cumulative variance")
        axis.set(title="PCA explained variance", ylabel="Share of total variance", ylim=(0, 1.08))
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.16)
        fig.tight_layout()
        fig.savefig(output_dir / "pca_variance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    report = {
        "fit_rows": len(train_frame),
        "seed": int(seed),
        "feature_columns": FEATURE_COLUMNS.copy(),
        "numeric_dimensions": 3,
        "price_used_as_input": False,
        "categorical_features_used": False,
        "preprocessing": {
            "fit_scope": "training listings only; no validation or test listings",
            "imputation": "per-feature training median",
            "imputer_statistics": dict(zip(FEATURE_COLUMNS, imputer.statistics_.tolist())),
            "missing_numeric_counts": {column: int(numeric[column].isna().sum()) for column in FEATURE_COLUMNS},
            "scaler": "StandardScaler: subtract training mean and divide by training population standard deviation",
            "scaler_mean": dict(zip(FEATURE_COLUMNS, scaler.mean_.tolist())),
            "scaler_scale": dict(zip(FEATURE_COLUMNS, scaler.scale_.tolist())),
        },
        "kmeans": {"candidate_k": list(range(2, 9)), "n_init": 20, "random_state": int(seed)},
        "selection_rule": "Highest training silhouette over k=2..8; exact ties choose smaller k.",
        "selected_k": int(selected_k),
        "selected_silhouette": best["silhouette"],
        "selection_scores": selection_rows,
        "cluster_profiles": _records(profiles),
        "scaled_centroids": _records(scaled_centroids),
        "original_centroids": _records(original_centroids),
        "profile_definition": "Cluster medians use observed original values; missing values are excluded, with observed counts reported. Price is added only after fitting for descriptive comparison.",
        "centroid_definition": "Centroids are means in scaled, imputed numeric space. The original-space table inverses scaling; expm1(log-mileage centroid) is not the arithmetic mean or median mileage.",
        "pca": {
            "input": "The same train-imputed and train-standardized three numeric features as KMeans.",
            "n_components": 3,
            "svd_solver": "full",
            "explained_variance_ratio": variance.tolist(),
            "cumulative_explained_variance_ratio": cumulative_variance.tolist(),
            "explained_variance": pca.explained_variance_.tolist(),
            "loadings": _records(loadings),
            "loading_definition": "Signed unit-length component coefficients (sklearn PCA.components_); not feature-component correlations. Component signs are arbitrary.",
        },
        "interpretation": [
            "Categories are deliberately omitted so the three numeric axes remain interpretable; brand, model and equipment differences are not represented.",
            "Silhouette is a geometric heuristic in this chosen standardized feature space; the winning k does not establish natural market segments or universal optimality.",
            "PCA is descriptive dimensionality reduction, not a supervised predictor or evidence of causal effects on price.",
            "The two-dimensional chart loses the variance in PC3; separation in the projection can differ from separation in all three dimensions.",
            "Median imputation and a single seed can influence clustering; no cluster stability study is claimed.",
            "Post-hoc price differences are descriptive asking-price summaries; price never enters clustering or PCA.",
        ],
        "files": {name: name for name in [
            "cluster_selection.csv", "cluster_selection.png", "cluster_profiles.csv",
            "cluster_centroids_scaled.csv", "cluster_centroids_original.csv",
            "cluster_assignments.csv", "pca_projection.csv", "pca_loadings.csv",
            "pca_variance.csv", "pca_variance.png", "pca_scatter.png", "unsupervised_report.json",
        ]},
    }
    with (output_dir / "unsupervised_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report
