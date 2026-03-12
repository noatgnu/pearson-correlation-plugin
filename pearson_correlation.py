"""Pearson correlation analysis with FDR BH correction and cluster detection."""

import os

import click
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def read_data_file(file_path: str) -> pd.DataFrame:
    """Read data file based on extension."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, sep=",")
    return pd.read_csv(file_path, sep="\t")


def get_sample_column_name(annotation: pd.DataFrame) -> str:
    """Find the Sample column in the annotation dataframe."""
    for col in annotation.columns:
        if col.lower() == 'sample':
            return col
    raise ValueError("Annotation file must have a 'Sample' column")


def impute_data(
    data: pd.DataFrame,
    method: str,
    knn_neighbors: int = 5
) -> pd.DataFrame:
    """
    Impute missing values in the data using sklearn imputers.

    Args:
        data: Data matrix with proteins as rows and samples as columns.
        method: Imputation method ('none', 'mean', 'median', 'zero', 'knn').
        knn_neighbors: Number of neighbors for KNN imputation.

    Returns:
        DataFrame with imputed values.

    Raises:
        ValueError: If method is 'none' and NaN values exist.
    """
    if method == "none":
        if data.isna().any().any():
            raise ValueError(
                "Data contains NaN values. Please select an imputation method or provide pre-imputed data."
            )
        return data

    if method == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    elif method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "median":
        imputer = SimpleImputer(strategy="median")
    elif method == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    imputed_values = imputer.fit_transform(data.values)
    return pd.DataFrame(imputed_values, index=data.index, columns=data.columns)


def calculate_pearson_correlation(
    data: pd.DataFrame,
    annotation: pd.DataFrame,
    index_col: str,
    target_col: str,
    imputation: str = "none",
    log2_transform: bool = False,
    knn_neighbors: int = 5
) -> pd.DataFrame:
    """
    Calculate Pearson correlation for each protein with a target column.

    Args:
        data: Data matrix with proteins as rows.
        annotation: Annotation dataframe with Sample and target column.
        index_col: Name of the protein identifier column.
        target_col: Target column name in annotation.
        imputation: Imputation method.
        log2_transform: Whether to apply log2 transformation.
        knn_neighbors: Number of neighbors for KNN imputation.

    Returns:
        DataFrame with Pearson correlation results.
    """
    sample_column_name = get_sample_column_name(annotation)

    if target_col not in annotation.columns:
        raise ValueError(f"Target column '{target_col}' not found in annotation file")

    annotation[target_col] = pd.to_numeric(annotation[target_col], errors='coerce')

    sample_columns = annotation[sample_column_name].tolist()
    sample_to_target = dict(zip(annotation[sample_column_name], annotation[target_col]))

    valid_samples = []
    targets = []
    for sample in sample_columns:
        if sample in data.columns and pd.notna(sample_to_target.get(sample)):
            valid_samples.append(sample)
            targets.append(sample_to_target[sample])

    if len(valid_samples) < 3:
        raise ValueError(
            f"Not enough valid samples with target values. Found {len(valid_samples)}, need at least 3"
        )

    targets = np.array(targets)

    data = data.set_index(index_col)
    data_subset = data[valid_samples].copy()

    if log2_transform:
        data_subset = np.log2(data_subset.replace(0, np.nan))
        data_subset.replace([np.inf, -np.inf], np.nan, inplace=True)

    data_subset = impute_data(data_subset, imputation, knn_neighbors)

    data_matrix = data_subset.values.astype(float)
    n_samples = len(targets)

    mean_x = data_matrix.mean(axis=1, keepdims=True)
    mean_y = targets.mean()

    x_centered = data_matrix - mean_x
    y_centered = targets - mean_y

    numerator = (x_centered * y_centered).sum(axis=1)
    denominator = np.sqrt((x_centered ** 2).sum(axis=1) * (y_centered ** 2).sum())

    with np.errstate(divide='ignore', invalid='ignore'):
        correlations = numerator / denominator

    t_stat = correlations * np.sqrt((n_samples - 2) / (1 - correlations ** 2))
    p_values = 2 * stats.t.sf(np.abs(t_stat), n_samples - 2)

    results = pd.DataFrame({
        'Protein': data_subset.index,
        'Correlation': correlations,
        'P_Value': p_values,
        'N_Samples': n_samples
    })

    return results


def apply_fdr_correction(
    results: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction.

    Args:
        results: DataFrame with P_Value column.
        alpha: Significance threshold.

    Returns:
        DataFrame with Q_Value and Significant columns added.
    """
    results = results.copy()

    valid_mask = ~results['P_Value'].isna()
    valid_pvalues = results.loc[valid_mask, 'P_Value'].values

    if len(valid_pvalues) == 0:
        results['Q_Value'] = np.nan
        results['Significant'] = False
        return results

    rejected, qvalues, _, _ = multipletests(
        valid_pvalues,
        alpha=alpha,
        method='fdr_bh'
    )

    results['Q_Value'] = np.nan
    results['Significant'] = False
    results.loc[valid_mask, 'Q_Value'] = qvalues
    results.loc[valid_mask, 'Significant'] = rejected

    return results


def compute_elbow_inertias(values: np.ndarray, max_k: int = 10):
    """Compute inertia values for different k values for elbow analysis."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))

    max_k = min(max_k, len(values) - 1)
    k_values = list(range(2, max_k + 1))
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_values)
        inertias.append(kmeans.inertia_)

    return k_values, inertias


def find_elbow_point(k_values: list, inertias: list) -> int:
    """Find the optimal k using the elbow method."""
    if len(k_values) < 3:
        return k_values[0]

    k_arr = np.array(k_values)
    inertia_arr = np.array(inertias)

    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    inertia_norm = (inertia_arr - inertia_arr.min()) / (inertia_arr.max() - inertia_arr.min() + 1e-10)

    distances = []
    p1 = np.array([k_norm[0], inertia_norm[0]])
    p2 = np.array([k_norm[-1], inertia_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    for i in range(len(k_values)):
        point = np.array([k_norm[i], inertia_norm[i]])
        dist = np.abs(line_vec[0] * (p1[1] - point[1]) - line_vec[1] * (p1[0] - point[0])) / (line_len + 1e-10)
        distances.append(dist)

    optimal_idx = np.argmax(distances)
    return k_values[optimal_idx]


def detect_clusters(
    correlations: np.ndarray,
    method: str,
    n_clusters: int = 5,
    eps: float = 0.5,
    min_samples: int = 5,
    auto_k: bool = False,
    max_k: int = 10
):
    """Detect clusters from correlation values."""
    if method == "none":
        return None, None

    valid_mask = ~np.isnan(correlations)
    valid_correlations = correlations[valid_mask]

    if len(valid_correlations) < 3:
        return None, None

    scaler = StandardScaler()
    scaled_correlations = scaler.fit_transform(valid_correlations.reshape(-1, 1))
    optimal_k = None

    if method == "kmeans":
        if auto_k:
            k_values, inertias = compute_elbow_inertias(valid_correlations, max_k)
            optimal_k = find_elbow_point(k_values, inertias)
            n_clusters = optimal_k
            print(f"[AUTO] Optimal number of clusters determined: {optimal_k}")

        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        valid_labels = clusterer.fit_predict(scaled_correlations)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        valid_labels = clusterer.fit_predict(scaled_correlations)
    else:
        return None, None

    labels = np.full(len(correlations), -1)
    labels[valid_mask] = valid_labels

    return labels, optimal_k


def generate_volcano_plot(
    results: pd.DataFrame,
    output_dir: str,
    alpha: float = 0.05
):
    """Generate volcano plot (correlation vs -log10 p-value)."""
    plot_df = results.copy()
    plot_df['neg_log10_pvalue'] = -np.log10(plot_df['P_Value'].clip(lower=1e-300))

    plot_df['Status'] = np.where(
        plot_df['Significant'],
        np.where(plot_df['Correlation'] > 0, 'Positive', 'Negative'),
        'Not Significant'
    )

    color_map = {
        'Positive': '#e74c3c',
        'Negative': '#3498db',
        'Not Significant': '#95a5a6'
    }

    fig = px.scatter(
        plot_df,
        x='Correlation',
        y='neg_log10_pvalue',
        color='Status',
        color_discrete_map=color_map,
        hover_name='Protein',
        hover_data={
            'Correlation': ':.4f',
            'P_Value': ':.2e',
            'Q_Value': ':.2e',
            'Status': False
        },
        title='Volcano Plot: Pearson Correlation'
    )

    fig.add_hline(
        y=-np.log10(alpha),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"p = {alpha}"
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        xaxis_title='Pearson Correlation',
        yaxis_title='-log₁₀(P-value)',
        template='plotly_white',
        width=900,
        height=700
    )

    fig.update_traces(marker=dict(size=6, opacity=0.7))

    fig.write_html(os.path.join(output_dir, "volcano_plot.html"))


def generate_cluster_plot(
    results: pd.DataFrame,
    output_dir: str
):
    """Generate scatter plot of correlations colored by cluster."""
    if 'Cluster' not in results.columns:
        return

    plot_df = results.copy()
    plot_df['neg_log10_pvalue'] = -np.log10(plot_df['P_Value'].clip(lower=1e-300))

    fig = px.scatter(
        plot_df,
        x='Correlation',
        y='neg_log10_pvalue',
        color='Cluster',
        hover_name='Protein',
        hover_data={
            'Correlation': ':.4f',
            'P_Value': ':.2e',
            'Q_Value': ':.2e',
            'Cluster': True
        },
        title='Cluster Plot: Proteins by Correlation'
    )

    fig.update_layout(
        xaxis_title='Pearson Correlation',
        yaxis_title='-log₁₀(P-value)',
        template='plotly_white',
        width=900,
        height=700
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))

    fig.write_html(os.path.join(output_dir, "cluster_plot.html"))


@click.command()
@click.option("--input_file", "-i", required=True, help="Path to the input data file")
@click.option("--annotation_file", "-a", required=True, help="Path to the annotation file")
@click.option("--index_col", "-x", required=True, help="Name of the protein identifier column")
@click.option("--target_col", "-t", required=True, help="Target column name from annotation")
@click.option("--imputation", type=click.Choice(["none", "mean", "median", "zero", "knn"]), default="none", help="Imputation method")
@click.option("--knn_neighbors", type=int, default=5, help="Number of neighbors for KNN imputation")
@click.option("--alpha", type=float, default=0.05, help="Significance threshold for FDR")
@click.option("--log2_transform", "-l", is_flag=True, help="Apply log2 transformation")
@click.option("--cluster_method", "-m", type=click.Choice(["none", "kmeans", "dbscan"]), default="none", help="Clustering method")
@click.option("--n_clusters", "-k", type=int, default=5, help="Number of clusters for KMeans")
@click.option("--dbscan_eps", "-e", type=float, default=0.5, help="DBSCAN epsilon parameter")
@click.option("--dbscan_min_samples", "-s", type=int, default=5, help="DBSCAN minimum samples")
@click.option("--auto_k", is_flag=True, help="Auto-determine optimal k using elbow method")
@click.option("--max_k", type=int, default=10, help="Maximum k to test for elbow method")
@click.option("--output_dir", "-o", required=True, help="Output directory")
def main(
    input_file: str,
    annotation_file: str,
    index_col: str,
    target_col: str,
    imputation: str,
    knn_neighbors: int,
    alpha: float,
    log2_transform: bool,
    cluster_method: str,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    auto_k: bool,
    max_k: int,
    output_dir: str
):
    """Calculate Pearson correlation with FDR BH correction."""
    os.makedirs(output_dir, exist_ok=True)

    data = read_data_file(input_file)
    annotation = read_data_file(annotation_file)

    results = calculate_pearson_correlation(
        data=data,
        annotation=annotation,
        index_col=index_col,
        target_col=target_col,
        imputation=imputation,
        log2_transform=log2_transform,
        knn_neighbors=knn_neighbors
    )

    results = apply_fdr_correction(results, alpha=alpha)

    if cluster_method != "none":
        cluster_labels, _ = detect_clusters(
            results['Correlation'].values,
            method=cluster_method,
            n_clusters=n_clusters,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            auto_k=auto_k,
            max_k=max_k
        )
        if cluster_labels is not None:
            results['Cluster'] = [
                f"Cluster_{label}" if label >= 0 else "Noise"
                for label in cluster_labels
            ]

    generate_volcano_plot(results, output_dir, alpha=alpha)
    if cluster_method != "none":
        generate_cluster_plot(results, output_dir)

    results = results.sort_values('P_Value', ascending=True)

    results.to_csv(
        os.path.join(output_dir, "correlation_results.tsv"),
        sep="\t",
        index=False
    )

    significant = results[results['Significant'] == True].copy()
    significant.to_csv(
        os.path.join(output_dir, "significant_correlations.tsv"),
        sep="\t",
        index=False
    )

    print(f"Total proteins analyzed: {len(results)}")
    print(f"Significant correlations (FDR < {alpha}): {len(significant)}")
    if len(significant) > 0:
        print(f"  Positive: {(significant['Correlation'] > 0).sum()}")
        print(f"  Negative: {(significant['Correlation'] < 0).sum()}")


if __name__ == "__main__":
    main()
