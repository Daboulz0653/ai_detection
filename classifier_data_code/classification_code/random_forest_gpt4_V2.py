

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import json
import random
import numpy as np

train_dataset = "GPA2Bench"
test_dataset = "GPA2Bench"

train_df = pd.read_csv(f"{test_dataset}/{test_dataset}_feature_matrix.csv")

test_df = pd.read_csv(f"{test_dataset}/{test_dataset}_feature_matrix.csv")
# test_df[""] = test_df.index
# test_df["model"] = "gpt-3.5-turbo"
# test_df.rename(columns={"source":"domain"}, inplace=True)
# test_df.to_csv(f"{test_dataset}/{test_dataset}_feature_matrix.csv")


for i in range(100):
    
    print("Creating random sample")

    train_sample_list = []

    for model in train_df['label'].unique():
        if model == 'authentic':
            for domain in train_df['domain'].unique():
                group_df = train_df[(train_df['label'] == model) & (train_df['domain'] == domain)]
                sample = group_df.sample(min(len(group_df), 1200))
                train_sample_list.extend([sample])
        else:
            for domain in train_df['domain'].unique():
                group_df = train_df[(train_df['label'] == model) & (train_df['domain'] == domain)]
                sample = group_df.sample(min(len(group_df), 1200))
                train_sample_list.extend([sample])

    # Concatenate all samples into a single dataframe
    train_sample_df = pd.concat(train_sample_list).reset_index(drop=True)

    train_size = len(train_sample_df)

    test_sample_list = []

    for model in test_df['label'].unique():
        if model == 'authentic':
            for domain in test_df['domain'].unique():
                group_df = test_df[(test_df['label'] == model) & (test_df['domain'] == domain)]
                sample = group_df.sample(min(len(group_df), 300))
                test_sample_list.extend([sample])
        else:
            for domain in test_df['domain'].unique():
                group_df = test_df[(test_df['label'] == model) & (test_df['domain'] == domain)]
                sample = group_df.sample(min(len(group_df), 300))
                test_sample_list.extend([sample])

    # Concatenate all samples into a single dataframe
    test_sample_df = pd.concat(test_sample_list).reset_index(drop=True)

    test_size = len(test_sample_df)

    sample_size = len(train_sample_df) + len(test_sample_df)

   

    print(f"Creating classifier {i}")

    feature_cols = ['mean_sen_len', 'sentiment',
       'male_pronouns', 'female_pronouns', 'TTR', 'lex_density',
       'its', 'during', 'between', 'how', 'see', 'be',
       "isn't", 'off', 'must', 'but', "couldn't", 'ours', 'a', 'about',
       'all', 'any', 'i', 'our', 'here', "aren't", 'and', 'ourselves',
       'itself', 'on', 'under', 'one', "you'll", 'too', 'this', 'after',
       'then', 'should', "that'll", 'me', 'why', 'your', 'until', "won't",
       'further', 'you', "needn't", "mightn't", 'they', "don't", 'each',
       'same', 'had', 'just', "wouldn't", 'my', 'into', 'that', 'are',
       "you've", 'than', 'do', 'as', 'the', 'them', 'there', 'does',
       'some', 'themselves', "weren't", "you'd", 'through', 'below', 'in',
       'don', "wasn't", 'an', 'were', 'for', 'has', 'very', 'before',
       'or', 'what', "hadn't", "doesn't", "shan't", 'having', 'no', 'not',
       'well', 'will', 'over', 'which', 'yourselves', 'once', 'am',
       'above', 'of', 'other', "it's", 'is', 'have', 'much', 'out',
       'would', 'by', 'again', "hasn't", 'myself', 'down', 'could',
       'theirs', 'from', 'while', 'with', 'who', 'against', 'doing',
       "you're", "mustn't", 'it', "shouldn't", "haven't", 'most', 'whom',
       'can', 'at', 'been', 'those', 'being', 'when', 'where', 'their',
       'was', 'never', 'nor', 'these', 'did', 'we', 'such', 'because',
       'up', 'few', 'more', 'made', 'yours', "should've", 'go', 'to',
       'yourself', 'only', 'so', 'might', 'own', 'now', 'if', 'upon',
       "didn't", 'both', 'say', 'said', 'ask', 'asked', 'reply ',
       'replied', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN',
       'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'PRP',
       'PRP$', 'IN', 'DT']
    X_train = train_sample_df[feature_cols]
    y_train = train_sample_df.label

    X_test = test_sample_df[feature_cols]
    y_test = test_sample_df.label

    # X = sample_df[feature_cols]
    # y = sample_df.label


    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Get precision, recall, and F1 scores for each category and overall weighted scores
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Add the predicted labels to the test set for comparison
    X_test = X_test.copy()  
    X_test['actual_label'] = y_test
    X_test['predicted_label'] = y_pred

    # Join with the original dataframe to include `id` and `author` columns
    misclassified = X_test[X_test['actual_label'] != X_test['predicted_label']]
    misclassified = misclassified.join(test_sample_df[['id']], how='left')

    # Display the misclassified rows with the required columns
    misclassified_result = misclassified[['id', 'actual_label', 'predicted_label']].sort_values(by=[ 'actual_label'])

    # Initialize dictionaries to store mislabel counts
    authentic_mislabeled =  misclassified_result[misclassified_result['actual_label'] == 'authentic']['id'].tolist()
    synthetic_mislabeled =  misclassified_result[misclassified_result['actual_label'] == 'synthetic']['id'].tolist()

    # Create dataframe to store feature importance scores
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })

    # Sort features by importance
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    top_10_df = feature_importance[:10]
    top_10_df.head(10)

    top_10_features = dict(zip(top_10_df['feature'], top_10_df['importance']))

    # Printing results to csv

    print(f"Saving results for classifier {i}")

    results_df = pd.DataFrame(columns=['accuracy',
                                                
                                            'Overall F1', 
                                            'Overall Precision',
                                            'Overall Recall', 
                                                
                                            'Authentic F1',
                                            'Authentic Precision',
                                            'Authentic Recall',
                                                
                                            'Synthetic F1',
                                            'Synthetic Precision',
                                            'Synthetic Recall',
                                                
                                            'authentic_mislabeled', 
                                            'synthetic_mislabeled', 
                                            'top_10_features'])

    results_df.at[0, 'accuracy'] = accuracy

    results_df.at[0, 'Overall F1'] = overall_f1
    results_df.at[0, 'Overall Precision'] = overall_precision
    results_df.at[0, 'Overall Recall'] = overall_recall

    results_df.at[0, 'Authentic F1'] = round(f1[0], 5)
    results_df.at[0, 'Authentic Precision'] = round(precision[0])
    results_df.at[0, 'Authentic Recall'] = recall[0]

    results_df.at[0, 'Synthetic F1'] = f1[1]
    results_df.at[0, 'Synthetic Precision'] = precision[1]
    results_df.at[0, 'Synthetic Recall'] = recall[1]

    results_df.at[0, 'authentic_mislabeled'] = authentic_mislabeled
    results_df.at[0, 'synthetic_mislabeled'] = synthetic_mislabeled
    results_df.at[0, 'top_10_features'] = top_10_features

    # Check if the file exists
    file_path = f"{test_dataset}/{test_dataset}_random_forest_final.csv"

    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(file_path, mode='w', header=True, index=False)


    run_info = { "user": "Zeina Daboul",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id": f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(0, 999)}",
                "sample_size": sample_size,
                "training_size": train_size,
                "test_size": test_size,
                "training_data": train_dataset,
                "testing_data": test_dataset,
                "training_models": train_sample_df['model'].unique(),
                "testing_models": test_sample_df['model'].unique(),
                "training_domains": train_sample_df['domain'].unique(),
                "testing_domains": test_sample_df['domain'].unique(),
                "accuracy": accuracy,
                "overall_f1": overall_f1,
                "overall_precision": overall_precision,
                "overall_recall": overall_recall,
                "authentic_f1": round(f1[0], 5),
                "authentic_precision": round(precision[0]),
                "authentic_recall": recall[0],
                "synthetic_f1": f1[1],
                "synthetic_precision": precision[1],
                "synthetic_recall": recall[1],
                "authentic_mislabeled": authentic_mislabeled,
                "synthetic_mislabeled": synthetic_mislabeled,
                "top_10_features": json.dumps(top_10_features) }

    run_info_df = pd.DataFrame([run_info])
    run_info_df.to_csv('classifier_log.csv', mode='a', index=False, header=False)



    print(f"Run {i} complete. Starting next run.")




