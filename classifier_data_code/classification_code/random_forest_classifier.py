

import pandas as pd
import os
import random
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# Configuration
CONFIG = {
    'text_model': 'gpt4',
    'n_runs': 100,
    'sample_size_per_category': 100,
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'data_path': "data/master_feature_matrix.csv",
    'output_dir': "data/"
}

# specify which subset of synthetic text we want to test the classifier on by model (e.g. gpt4, gpt3.5, etc.)
text_model = CONFIG['text_model']

# Define the features you want to use for the classification (using column headers from input file)
feature_cols = ['mean_sen_len', 'sentiment',
       'male_pronouns', 'female_pronouns', 'TTR', 'lex_density',
       'concreteness', 'its', 'during', 'between', 'how', 'see', 'be',
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


# Load in the master feature matrix with all of the features for all models and authors
master_feature_matrix_df = pd.read_csv(CONFIG['data_path'])

# Filter the dataframe to only include the specified model text and the human/authentic text
df = master_feature_matrix_df[(master_feature_matrix_df['model'] == 'authentic') | (master_feature_matrix_df['model'] == text_model)]

# Generate a random 5-digit number for the output filename
random_suffix = random.randint(10000, 99999)

# Start the classification process (this will run the classifier n_runs times, each time creating a new random sample)
for i in range(CONFIG['n_runs']):
    
    print(f"Creating random sample for run {i+1}/{CONFIG['n_runs']}")

    sample_list = []

    for author in df['author'].unique():
        # Check if we have enough samples for each category
        auth_samples = df[(df['author'] == author) & (df['category'] == 'authentic')]
        syn_samples = df[(df['author'] == author) & (df['category'] == 'synthetic')]
        
        if len(auth_samples) < CONFIG['sample_size_per_category']:
            print(f"Warning: Only {len(auth_samples)} authentic samples for {author}, using all available")
            sample_auth = auth_samples
        else:
            sample_auth = auth_samples.sample(CONFIG['sample_size_per_category'], random_state=CONFIG['random_state']+i)
            
        if len(syn_samples) < CONFIG['sample_size_per_category']:
            print(f"Warning: Only {len(syn_samples)} synthetic samples for {author}, using all available")
            sample_syn = syn_samples
        else:
            sample_syn = syn_samples.sample(CONFIG['sample_size_per_category'], random_state=CONFIG['random_state']+i+1000)
        
        sample_list.extend([sample_auth, sample_syn])

    # Concatenate all samples into a single dataframe
    sample_df = pd.concat(sample_list).reset_index(drop=True)

    # Get all sample IDs
    all_sample_ids = sample_df['id'].tolist()

    print(f"Creating classifier {i}")

    
    X = sample_df[feature_cols] 
    y = sample_df.category


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']+i)
    
    # Get train and test IDs
    train_ids = sample_df.iloc[X_train.index]['id'].tolist()
    test_ids = sample_df.iloc[X_test.index]['id'].tolist()

    # Initialize the Random Forest model
    classification_model = RandomForestClassifier(n_estimators=CONFIG['n_estimators'], random_state=CONFIG['random_state'])

    # Train the model
    classification_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classification_model.predict(X_test)

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
    misclassified = misclassified.join(sample_df[['id', 'author']], how='left')

    # Display the misclassified rows with the required columns
    misclassified_result = misclassified[['id', 'author', 'actual_label', 'predicted_label']].sort_values(by=['author', 'actual_label'])

    # Initialize dictionaries to store mislabel counts
    authentic_mislabeled =  misclassified_result[ misclassified_result['actual_label'] == 'authentic']['id'].tolist()
    synthetic_mislabeled =  misclassified_result[ misclassified_result['actual_label'] == 'synthetic']['id'].tolist()

    # Create dataframe to store feature importance scores
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': classification_model.feature_importances_
    })

    # Sort features by importance
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    top_10_df = feature_importance[:10]
    top_10_df.head(10)

    top_10_features = dict(zip(top_10_df['feature'], top_10_df['importance']))

    # Printing results to csv

    print(f"Saving results for classifier {i}")

    results_df = pd.DataFrame(columns=['run_date',
                                            'features_used',
                                            'accuracy',
                                                
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
                                            'top_10_features',
                                            'all_sample_ids',
                                            'train_ids',
                                            'test_ids'])

    results_df.at[0, 'run_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_df.at[0, 'features_used'] = feature_cols
    results_df.at[0, 'accuracy'] = accuracy

    results_df.at[0, 'Overall F1'] = overall_f1
    results_df.at[0, 'Overall Precision'] = overall_precision
    results_df.at[0, 'Overall Recall'] = overall_recall

    results_df.at[0, 'Authentic F1'] = round(f1[0], 5)
    results_df.at[0, 'Authentic Precision'] = round(precision[0], 5)
    results_df.at[0, 'Authentic Recall'] = round(recall[0], 5)

    results_df.at[0, 'Synthetic F1'] = round(f1[1], 5)
    results_df.at[0, 'Synthetic Precision'] = round(precision[1], 5)
    results_df.at[0, 'Synthetic Recall'] = round(recall[1], 5)

    results_df.at[0, 'authentic_mislabeled'] = authentic_mislabeled
    results_df.at[0, 'synthetic_mislabeled'] = synthetic_mislabeled
    results_df.at[0, 'top_10_features'] = top_10_features
    results_df.at[0, 'all_sample_ids'] = all_sample_ids
    results_df.at[0, 'train_ids'] = train_ids
    results_df.at[0, 'test_ids'] = test_ids


    # Check if the file exists
    file_path = f"{CONFIG['output_dir']}random_forest_classifier_results_{random_suffix}.csv"

    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(file_path, mode='w', header=True, index=False)

    print(f"Run {i+1} complete. Starting next run.")




