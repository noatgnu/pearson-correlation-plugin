"""Pearson correlation analysis with FDR BH correction."""

import os

import click
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
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
    knn_neighbors: int = 5,
    sample_indices: list = None
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
        sample_indices: Optional list of sample names to use (for group-based analysis).

    Returns:
        DataFrame with Pearson correlation results.
    """
    sample_column_name = get_sample_column_name(annotation)

    if target_col not in annotation.columns:
        raise ValueError(f"Target column '{target_col}' not found in annotation file")

    annotation = annotation.copy()
    annotation[target_col] = pd.to_numeric(annotation[target_col], errors='coerce')

    if sample_indices is not None:
        annotation = annotation[annotation[sample_column_name].isin(sample_indices)]

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

    if index_col in data.columns:
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


def generate_volcano_plot(
    results: pd.DataFrame,
    output_dir: str,
    alpha: float = 0.05,
    suffix: str = "",
    title_suffix: str = ""
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

    title = 'Volcano Plot: Pearson Correlation'
    if title_suffix:
        title = f'{title} - {title_suffix}'

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
        title=title
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
        yaxis_title='-log10(P-value)',
        template='plotly_white',
        width=900,
        height=700
    )

    fig.update_traces(marker=dict(size=6, opacity=0.7))

    filename = f"volcano_plot{suffix}.html" if suffix else "volcano_plot.html"
    fig.write_html(os.path.join(output_dir, filename))


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


@click.command()
@click.option("--input_file", "-i", required=True, help="Path to the input data file")
@click.option("--annotation_file", "-a", required=True, help="Path to the annotation file")
@click.option("--index_col", "-x", required=True, help="Name of the protein identifier column")
@click.option("--target_col", "-t", required=True, help="Target column name from annotation")
@click.option("--grouping_col", "-g", default="", help="Column to group samples for separate volcano plots")
@click.option("--imputation", type=click.Choice(["none", "mean", "median", "zero", "knn"]), default="none", help="Imputation method")
@click.option("--knn_neighbors", type=int, default=5, help="Number of neighbors for KNN imputation")
@click.option("--alpha", type=float, default=0.05, help="Significance threshold for FDR")
@click.option("--log2_transform", "-l", is_flag=True, help="Apply log2 transformation")
@click.option("--output_dir", "-o", required=True, help="Output directory")
def main(
    input_file: str,
    annotation_file: str,
    index_col: str,
    target_col: str,
    grouping_col: str,
    imputation: str,
    knn_neighbors: int,
    alpha: float,
    log2_transform: bool,
    output_dir: str
):
    """Calculate Pearson correlation with FDR BH correction."""
    os.makedirs(output_dir, exist_ok=True)

    data = read_data_file(input_file)
    annotation = read_data_file(annotation_file)
    sample_column_name = get_sample_column_name(annotation)

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

    generate_volcano_plot(results, output_dir, alpha=alpha)

    if grouping_col and grouping_col in annotation.columns:
        groups = annotation[grouping_col].dropna().unique()
        print(f"Generating volcano plots for {len(groups)} groups in '{grouping_col}'")

        for group in groups:
            group_samples = annotation[annotation[grouping_col] == group][sample_column_name].tolist()

            try:
                group_results = calculate_pearson_correlation(
                    data=data,
                    annotation=annotation,
                    index_col=index_col,
                    target_col=target_col,
                    imputation=imputation,
                    log2_transform=log2_transform,
                    knn_neighbors=knn_neighbors,
                    sample_indices=group_samples
                )
                group_results = apply_fdr_correction(group_results, alpha=alpha)

                suffix = f"_{sanitize_filename(group)}"
                generate_volcano_plot(
                    group_results,
                    output_dir,
                    alpha=alpha,
                    suffix=suffix,
                    title_suffix=str(group)
                )

                group_results['Group'] = group
                group_results.to_csv(
                    os.path.join(output_dir, f"correlation_results_{sanitize_filename(group)}.tsv"),
                    sep="\t",
                    index=False
                )
                print(f"  Group '{group}': {len(group_samples)} samples, {(group_results['Significant']).sum()} significant")

            except ValueError as e:
                print(f"  Group '{group}': Skipped - {e}")

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
