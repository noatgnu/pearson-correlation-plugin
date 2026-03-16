"""Pearson correlation analysis with FDR BH correction."""

import os

import click
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        DataFrame with (potentially) imputed values.
    """
    if method == "none":
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

    X = data_subset.values.astype(float)
    y = targets.astype(float)
    proteins = data_subset.index.tolist()
    n_proteins = X.shape[0]
    n_samples = X.shape[1]

    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]
    X_valid = X[:, valid_mask]
    n_valid = valid_mask.sum()

    if n_valid < 3:
        return pd.DataFrame({
            'Protein': proteins,
            'Correlation': [np.nan] * n_proteins,
            'P_Value': [np.nan] * n_proteins,
            'N_Samples': [n_valid] * n_proteins
        })

    y_mean = np.mean(y_valid)
    y_centered = y_valid - y_mean
    y_std = np.std(y_valid, ddof=0)

    X_mean = np.mean(X_valid, axis=1, keepdims=True)
    X_centered = X_valid - X_mean
    X_std = np.std(X_valid, axis=1, ddof=0)

    valid_std = X_std > 0
    correlations = np.full(n_proteins, np.nan)
    correlations[valid_std] = np.sum(X_centered[valid_std] * y_centered, axis=1) / (n_valid * X_std[valid_std] * y_std)
    correlations = np.clip(correlations, -1.0, 1.0)

    pvalues = np.full(n_proteins, np.nan)
    valid_corr = valid_std & ~np.isnan(correlations)
    if np.any(valid_corr):
        r_valid = correlations[valid_corr]
        t_stat = r_valid * np.sqrt((n_valid - 2) / (1 - r_valid**2 + 1e-10))
        pvalues[valid_corr] = 2 * stats.t.sf(np.abs(t_stat), df=n_valid - 2)

    return pd.DataFrame({
        'Protein': proteins,
        'Correlation': correlations,
        'P_Value': pvalues,
        'N_Samples': [n_valid] * n_proteins
    })


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
        DataFrame with Adjusted_P_Value and Significant columns added.
    """
    results = results.copy()

    valid_mask = ~results['P_Value'].isna()
    valid_pvalues = results.loc[valid_mask, 'P_Value'].values

    if len(valid_pvalues) == 0:
        results['Adjusted_P_Value'] = np.nan
        results['Significant'] = False
        return results

    rejected, adj_pvalues, _, _ = multipletests(
        valid_pvalues,
        alpha=alpha,
        method='fdr_bh'
    )

    results['Adjusted_P_Value'] = np.nan
    results['Significant'] = False
    results.loc[valid_mask, 'Adjusted_P_Value'] = adj_pvalues
    results.loc[valid_mask, 'Significant'] = rejected

    return results


def generate_scatter_plots(
    data: pd.DataFrame,
    annotation: pd.DataFrame,
    results: pd.DataFrame,
    index_col: str,
    target_col: str,
    output_dir: str,
    top_n: int = 9
):
    """Generate scatter plots for top significant proteins."""
    sig_results = results[results['Significant'] == True].nsmallest(top_n, 'Adjusted_P_Value')

    if len(sig_results) == 0:
        return

    sample_col = None
    for col in annotation.columns:
        if col.lower() == 'sample':
            sample_col = col
            break

    if sample_col is None:
        return

    if index_col in data.columns:
        data = data.set_index(index_col)

    n_plots = len(sig_results)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[str(p)[:30] for p in sig_results['Protein'].tolist()]
    )

    for idx, (_, row) in enumerate(sig_results.iterrows()):
        protein = row['Protein']
        r_val = row['Correlation']
        adj_p_val = row['Adjusted_P_Value']
        r_idx = idx // n_cols + 1
        c_idx = idx % n_cols + 1

        if protein not in data.index:
            continue

        protein_values = data.loc[protein]

        plot_data = []
        for _, ann_row in annotation.iterrows():
            sample = ann_row[sample_col]
            if sample in protein_values.index:
                val = protein_values[sample]
                target_val = ann_row[target_col]
                if pd.notna(val) and pd.notna(target_val):
                    plot_data.append({'abundance': val, 'target': float(target_val)})

        if not plot_data:
            continue

        plot_df = pd.DataFrame(plot_data)

        fig.add_trace(
            go.Scatter(
                x=plot_df['target'],
                y=plot_df['abundance'],
                mode='markers',
                marker=dict(size=8, opacity=0.7, color='#3498db'),
                name=str(protein)[:20],
                showlegend=False,
                hovertemplate=f'r={r_val:.3f}, Adjusted p={adj_p_val:.2e}<extra></extra>'
            ),
            row=r_idx, col=c_idx
        )

        if len(plot_df) >= 2:
            z = np.polyfit(plot_df['target'], plot_df['abundance'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_df['target'].min(), plot_df['target'].max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=p(x_line),
                    mode='lines',
                    line=dict(color='#e74c3c', width=2),
                    showlegend=False
                ),
                row=r_idx, col=c_idx
            )

    fig.update_layout(
        title=f'Top {len(sig_results)} Significant Correlations (Adjusted P-values)',
        template='plotly_white',
        height=350 * n_rows,
        width=350 * n_cols
    )

    fig.write_html(os.path.join(output_dir, 'scatter_plots.html'))


def generate_ranked_bar_plot(
    results: pd.DataFrame,
    output_dir: str,
    top_n: int = 30
):
    """Generate ranked bar plot of top correlations."""
    valid_results = results.dropna(subset=['Correlation', 'P_Value'])

    pos_top = valid_results.nlargest(top_n // 2, 'Correlation')
    neg_top = valid_results.nsmallest(top_n // 2, 'Correlation')
    plot_df = pd.concat([pos_top, neg_top]).drop_duplicates()
    plot_df = plot_df.sort_values('Correlation', ascending=True)

    if len(plot_df) == 0:
        return

    colors = ['#e74c3c' if x > 0 else '#3498db' for x in plot_df['Correlation']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[str(p)[:40] for p in plot_df['Protein']],
        x=plot_df['Correlation'],
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}<br>r=%{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Top {len(plot_df)} Correlations (Positive and Negative)',
        xaxis_title='Pearson Correlation',
        yaxis_title='Protein',
        template='plotly_white',
        height=max(400, len(plot_df) * 25),
        width=800
    )

    fig.write_html(os.path.join(output_dir, 'ranked_correlations.html'))


def generate_volcano_plot(
    results: pd.DataFrame,
    output_dir: str,
    alpha: float = 0.05,
    suffix: str = "",
    title_suffix: str = "",
    use_adjusted_pvalue: bool = True
):
    """
    Generate volcano plot (correlation vs -log10 p-value).

    :param results: DataFrame with correlation results.
    :param output_dir: Output directory path.
    :param alpha: Significance threshold.
    :param suffix: Filename suffix.
    :param title_suffix: Title suffix for the plot.
    :param use_adjusted_pvalue: If True, use Adjusted_P_Value for cutoff; if False, use P_Value.
    """
    plot_df = results.copy()

    if use_adjusted_pvalue:
        pvalue_col = 'Adjusted_P_Value'
        y_label = '-log10(Adjusted P-value)'
        plot_type = 'adjusted'
        significance_col = plot_df['Adjusted_P_Value'].fillna(1)
    else:
        pvalue_col = 'P_Value'
        y_label = '-log10(P-value)'
        plot_type = 'raw'
        significance_col = plot_df['P_Value'].fillna(1)

    plot_df['neg_log10_pvalue'] = -np.log10(plot_df[pvalue_col].clip(lower=1e-300))

    plot_df['Status'] = np.where(
        significance_col < alpha,
        np.where(plot_df['Correlation'] > 0, 'Positive', 'Negative'),
        'Not Significant'
    )

    color_map = {
        'Positive': '#e74c3c',
        'Negative': '#3498db',
        'Not Significant': '#95a5a6'
    }

    pvalue_label = "Adjusted P-value" if use_adjusted_pvalue else "P-value"
    title = f'Volcano Plot: Pearson Correlation ({pvalue_label} cutoff)'
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
            'Adjusted_P_Value': ':.2e',
            'Status': False
        },
        title=title
    )

    fig.add_hline(
        y=-np.log10(alpha),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{pvalue_label} = {alpha}"
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        xaxis_title='Pearson Correlation',
        yaxis_title=y_label,
        template='plotly_white',
        width=900,
        height=700
    )

    fig.update_traces(marker=dict(size=6, opacity=0.7))

    if suffix:
        filename = f"volcano_plot_{plot_type}{suffix}.html"
    else:
        filename = f"volcano_plot_{plot_type}.html"
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
    
    # Filter annotation: only keep samples that have a value in the target column
    if target_col in annotation.columns:
        # Convert to numeric to find missing values
        annotation[target_col] = pd.to_numeric(annotation[target_col], errors='coerce')
        original_len = len(annotation)
        annotation = annotation.dropna(subset=[target_col])
        filtered_len = len(annotation)
        if original_len > filtered_len:
            print(f"Dropped {original_len - filtered_len} samples from annotation without values in '{target_col}'")

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

    generate_volcano_plot(results, output_dir, alpha=alpha, use_adjusted_pvalue=True)
    generate_volcano_plot(results, output_dir, alpha=alpha, use_adjusted_pvalue=False)
    generate_ranked_bar_plot(results, output_dir)
    generate_scatter_plots(data, annotation, results, index_col, target_col, output_dir)

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
                    title_suffix=str(group),
                    use_adjusted_pvalue=True
                )
                generate_volcano_plot(
                    group_results,
                    output_dir,
                    alpha=alpha,
                    suffix=suffix,
                    title_suffix=str(group),
                    use_adjusted_pvalue=False
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
