import pandas as pd
from collections import Counter

from IPython.lib.editorhooks import crimson_editor
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from statistics import mean
from collections import Counter
import ast
from collections import defaultdict


def accuracy_table(df, sub_folder):
    columns = [
        'Overall F1', 'Overall Precision', 'Overall Recall',
        'Authentic F1', 'Authentic Precision', 'Authentic Recall',
        'Synthetic F1', 'Synthetic Precision', 'Synthetic Recall'
    ]

    # Calculate mean and standard deviation
    mean_std_df = pd.DataFrame({
        'Metric': columns,
        'Mean': df[columns].mean().values,
        'Standard Deviation': df[columns].std().values})

    mean_std_df.to_csv(f"{sub_folder}/eval_metrics.csv", index=False)

def confusion(df, sub_folder):
    from collections import Counter
    import ast

    # Extract all ID numbers
    synthetic_ids = []

    for s in df['synthetic_mislabeled'].tolist():
        ids = ast.literal_eval(s)  # Convert string to list
        synthetic_ids.extend(ids)

    # Count occurrences
    synthetic_id_counts = Counter(synthetic_ids)

    # Create a dataframe
    synthetic_misclassified_df = pd.DataFrame(synthetic_id_counts.items(), columns=['id', 'Counts'])
    synthetic_misclassified_df = synthetic_misclassified_df.sort_values(by='Counts', ascending=False)

    # Extract all ID numbers
    authentic_ids = []
    for s in df['authentic_mislabeled'].tolist():
        ids = ast.literal_eval(s)  # Convert string to list
        authentic_ids.extend(ids)

    # Count occurrences
    authentic_id_counts = Counter(authentic_ids)

    # Create a dataframe
    authentic_misclassified_df = pd.DataFrame(authentic_id_counts.items(), columns=['id', 'Counts'])
    authentic_misclassified_df = authentic_misclassified_df.sort_values(by='Counts', ascending=False)

    FP = round((authentic_misclassified_df['Counts'].sum()))
    # False Positives
    FN = round((synthetic_misclassified_df['Counts'].sum()))  # False Negatives
    TN = 249800 - FN  # True Negatives
    TP = 249800 - FP

    # Create the confusion M4_matrix_final
    conf_matrix = np.array([[TP, FP], [FN, TN]])

    # Define class labels
    labels = ['Authentic', 'Synthetic']

    # Plot the confusion M4_matrix_final
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Aggregate over 100)')

    plt.savefig(f"{sub_folder}/classifier_random_confusion_matrix_aggregate.png")
    plt.close()

def feature_importance_scores(df, sub_folder):
    key_counts = defaultdict(int)
    key_scores = defaultdict(float)

    # Iterate through each row in df['top_10_features']
    for feature_str in df['top_10_features']:
        try:
            # Parse the string into a Python dictionary
            feature_dict = ast.literal_eval(feature_str)

            # Check if the parsed object is a dictionary
            if isinstance(feature_dict, dict):
                for key, score in feature_dict.items():
                    key_counts[key] += 1  # Count occurrences of each key
                    key_scores[key] += score  # Accumulate scores for each key
            else:
                print(f"Skipping non-dictionary entry: {feature_str}")
        except (ValueError, SyntaxError):
            print(f"Skipping invalid entry: {feature_str}")

    # Create the result dataframe
    result_df = pd.DataFrame({
        'Key': list(key_counts.keys()),
        'Count': list(key_counts.values()),
        'Mean Score': [key_scores[key] / key_counts[key] for key in key_counts]
    })

    result_df.sort_values(by='Count', inplace=True, ascending=False)
    result_df.to_csv(f"{sub_folder}/top_features_and_scores.csv")

def feature_box_plot(df, sub_folder):
    dict_list = []

    # Loop through the column, and check if the row is a dictionary or a string
    for index, row in df['top_10_features'].items():
        if isinstance(row, str):
            # Convert the string to a dictionary
            row_dict = ast.literal_eval(row)
        elif isinstance(row, dict):
            # If it's already a dictionary, use it directly
            row_dict = row
        else:
            # Skip if the row is neither a string nor a dictionary
            print(f"Row {index} is not a valid type, skipping.")
            continue

        # Append the dictionary to the list
        dict_list.append(row_dict)

    all_keys = []

    # Loop through each dictionary and extract the keys
    for d in dict_list:
        all_keys.extend(d.keys())

    # Use Counter to count the occurrences of each key
    key_counts = Counter(all_keys)

    # Convert the Counter dictionary to a DataFrame
    feature_keys_df = pd.DataFrame(list(key_counts.items()), columns=['Key', 'Count'])

    # Assuming dict_list is your list of dictionaries and key_counts is already populated
    # Step 2: Get the top 10 keys based on their occurrence
    top_keys = [k for k, v in key_counts.most_common(10)]  # Use `key_counts` to find top keys

    # Step 3: Create a dictionary to store values associated with each key
    key_values = {key: [] for key in top_keys}

    # Step 4: Collect values for each key across all dictionaries
    for d in dict_list:
        for key in top_keys:  # Use `top_keys` here
            if key in d:
                key_values[key].append(d[key])

    # Step 5: Convert key-values dictionary into a DataFrame for plotting
    data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in key_values.items()]))

    # Step 6: Order features by their median importance scores
    medians = data.median().sort_values(ascending=False)
    data = data[medians.index]  # Reorder DataFrame columns based on sorted medians

    # Step 7: Create a box plot for the top 10 features ordered by median
    plt.figure(figsize=(12, 8))  # Adjust the figure size if necessary
    sns.boxplot(data=data)

    # Customize the plot
    plt.ylim(0, 0.1)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('RANDOM FOREST: Box Plot of Top 10 Features Ordered by Median Importance Score')
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability

    # Display the plot and save the figure
    plt.tight_layout()
    plt.savefig(f"{sub_folder}/randomforest_topfeatures_box.jpg")
    plt.close()




# Store results as a list of dictionaries for easy conversion to DataFrame
results = []

def basic_stats(df, domain):
    accuracy_values = df['accuracy'].tolist()
    stats = {
        # 'model': model,
        'domain': domain,
        'mean': mean(accuracy_values),
        'min': min(accuracy_values),
        'max': max(accuracy_values),
        'std': df['accuracy'].std()
    }
    results.append(stats)


def main():
    dir_name = sys.argv[1]
    print(dir_name)
    for file in os.listdir(dir_name):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(dir_name, file)
        print(file_path)
        domain_sub_folder_name = "_".join(file.split("_")[:2])
        domain_sub_folder = os.path.join(dir_name, domain_sub_folder_name)
        model_sub_folder_name = "_".join(file.split("_")[2:3])
        model_sub_folder = os.path.join(domain_sub_folder, model_sub_folder_name)
        print(domain_sub_folder_name)
        print(model_sub_folder_name)

        # os.makedirs(domain_sub_folder, exist_ok=True)
        # os.makedirs(model_sub_folder, exist_ok=True)


        domain = "_".join(file.split("_")[:2])
        model = "_".join(file.split("_")[2:3])  # adjust as needed

        print("Processing group: " + domain_sub_folder_name)
        df = pd.read_csv(file_path, encoding="utf-8")
        accuracy_table(df, dir_name)
        confusion(df, dir_name)
        feature_importance_scores(df, dir_name)
        feature_box_plot(df, dir_name)
        basic_stats(df, domain)



# Convert results to DataFrame
    df_summary = pd.DataFrame(results)
    output_csv = os.path.join(dir_name, "summary_stats_long.csv")
    df_summary.to_csv(output_csv, index=False)
    print(f"Saved long-format summary to {output_csv}")


if __name__ == "__main__":
    main()

