# Pearson Correlation Analysis

**ID**: `pearson-correlation`  
**Version**: 1.0.0  
**Category**: statistics  
**Author**: CauldronGO Team

## Description

Calculate Pearson correlation between protein abundance and a target variable with FDR BH correction.

## Runtime

- **Environments**: `python`

- **Entrypoint**: `pearson_correlation.py`

## Inputs

| Name | Label | Type | Required | Default | Visibility |
|------|-------|------|----------|---------|------------|
| `input_file` | Input File | file | Yes | - | Always visible |
| `annotation_file` | Sample Annotation File | file | Yes | - | Always visible |
| `index_col` | Index Column | column-selector (single) | Yes | - | Always visible |
| `target_col` | Target Column | text | Yes | - | Always visible |
| `grouping_col` | Grouping Column | text | No |  | Always visible |
| `imputation` | Imputation Method | select (none, mean, median, zero, knn) | No | none | Always visible |
| `knn_neighbors` | KNN Neighbors | number (min: 1, max: 20, step: 1) | No | 5 | Visible when `imputation` = `knn` |
| `alpha` | Significance Level | number (min: 0, max: 0, step: 0) | No | 0.05 | Always visible |
| `log2_transform` | Log2 Transform Data | boolean | No | false | Always visible |

### Input Details

#### Input File (`input_file`)

Data matrix file with proteins as rows and samples as columns.


#### Sample Annotation File (`annotation_file`)

Sample annotation file containing Sample and target columns.


#### Index Column (`index_col`)

Column containing protein identifiers.

- **Column Source**: `input_file`

#### Target Column (`target_col`)

Numeric column name from annotation to correlate with.


#### Grouping Column (`grouping_col`)

Column from annotation to group samples for separate volcano plots per group.


#### Imputation Method (`imputation`)

Method for handling missing values. Select 'none' if data is pre-imputed.

- **Options**: `none`, `mean`, `median`, `zero`, `knn`

#### KNN Neighbors (`knn_neighbors`)

Number of neighbors for KNN imputation.


#### Significance Level (`alpha`)

Significance threshold for FDR correction.


#### Log2 Transform Data (`log2_transform`)

Apply log2 transformation to the data before correlation analysis.


## Outputs

| Name | File | Type | Format | Description |
|------|------|------|--------|-------------|
| `correlation_results` | `correlation_results.tsv` | data | tsv | Full correlation results with p-values and FDR-corrected q-values. |
| `significant_results` | `significant_correlations.tsv` | data | tsv | Filtered results showing only significant correlations. |
| `volcano_plot` | `volcano_plot.html` | html | html | Volcano plot showing correlation vs significance. |
| `scatter_plots` | `scatter_plots.html` | html | html | Scatter plots showing protein abundance vs target for top significant proteins. |
| `ranked_correlations` | `ranked_correlations.html` | html | html | Bar plot showing proteins ranked by correlation coefficient. |

## Sample Annotation

This plugin supports sample annotation:

- **Annotation File**: `annotation_file`

## Requirements

- **Python Version**: >=3.11

### Python Dependencies (External File)

Dependencies are defined in: `requirements.txt`

- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scipy>=1.10.0`
- `statsmodels>=0.14.0`
- `scikit-learn>=1.3.0`
- `plotly>=5.18.0`
- `click>=8.0.0`

> **Note**: When you create a custom environment for this plugin, these dependencies will be automatically installed.

## Example Data

This plugin includes example data for testing:

```yaml
  annotation_file: diann/annotation_with_score.txt
  index_col: Protein.Group
  target_col: Score
  grouping_col: Condition
  imputation: mean
  alpha: 0.05
  log2_transform: true
  input_file: diann/imputed.data.txt
```

Load example data by clicking the **Load Example** button in the UI.

## Usage

### Via UI

1. Navigate to **statistics** → **Pearson Correlation Analysis**
2. Fill in the required inputs
3. Click **Run Analysis**

### Via Plugin System

```typescript
const jobId = await pluginService.executePlugin('pearson-correlation', {
  // Add parameters here
});
```
