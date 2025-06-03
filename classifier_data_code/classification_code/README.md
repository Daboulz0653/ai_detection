# Random Forest Text Classifier

A Random Forest classifier for distinguishing between 'authentic' human-written text and 'synthetic' AI-generated text. This code performs 100 random forest classification runs with different random samples.

## k_fold_author_validation_classifier




## Requirements

```python
pandas
scikit-learn
```

## Input Data Format

The script expects a CSV file with the following columns:
- `id`: Unique identifier for each text sample
- `author`: Author name
- `model`: Either 'authentic' (human-written) or the specified AI model (e.g., 'gpt4')
- `category`: Either 'authentic' or 'synthetic'
- Feature columns: All the linguistic features listed in `feature_cols`


## Configuration

The `CONFIG` dictionary at the top of the script controls all experimental parameters:

```python
CONFIG = {
    'text_model': 'gpt4',                    # AI model to test against (e.g., 'gpt4', 'gpt3.5')
    'n_runs': 100,                           # Number of classification runs
    'sample_size_per_category': 100,         # Sample size per author per category
    'test_size': 0.2,                        # Proportion of data for testing (20%)
    'random_state': 42,                      # Base random seed for reproducibility
    'n_estimators': 100,                     # Number of trees in Random Forest
    'data_path': "data/master_feature_matrix.csv",  # Path to input data
    'output_dir': "data/"                    # Output directory for results
}
```


## Reproducibility

This code uses random_state assignments at various points to randomly sample subsets of the data, and also to control the random forest model initialization. Currently, the random_state assignment is set at 42. It is designed to remain 42 in the model initialization for all 100 classifier. However, the code **updates the random_state for each random sampling of text** by adding 1 to the random_state each time the classifier loops. We took this approach to ensure different random sampling of the texts across each run of the classifier, while preventing variation in model initialization, thereby ensuring the results for each run only vary because of the random sample of texts in the classifier, rather than the model itself.

## Output

The script generates a CSV file with a random 5-digit suffix: `random_forest_classifier_results_XXXXX.csv`

### Output Columns:
- `run_date`: Timestamp of when the run was executed
- `features_used`: Complete list of features used for classification
- `accuracy`: Overall classification accuracy
- `Overall F1/Precision/Recall`: Weighted average metrics across both categories
- `Authentic F1/Precision/Recall`: Metrics for authentic text classification
- `Synthetic F1/Precision/Recall`: Metrics for synthetic text classification
- `authentic_mislabeled`: List of IDs for authentic texts misclassified as synthetic
- `synthetic_mislabeled`: List of IDs for synthetic texts misclassified as authentic
- `top_10_features`: Dictionary of the 10 most important features and their importance scores
- `all_sample_ids`: Complete list of text IDs used in this run
- `train_ids`: List of text IDs used for training
- `test_ids`: List of text IDs used for testing

## Usage

1. **Prepare your data**: Ensure your CSV file matches the expected format
2. **Configure parameters**: Modify the `CONFIG` dictionary as needed
3. **Update features**: Change the features to specify only those features you want to use for classification. The features included need to be in the input data file. The features are passed as column headers.
4. **Run the script**:
   ```bash
   python random_forest_classifier.py
   ```
5. **Analyze results**: The output CSV contains comprehensive metrics for all runs


## Citation

When using code or data from this repository, please cite
