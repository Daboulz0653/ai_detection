#!/usr/bin/env python


import pandas as pd
import os
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import random
from datetime import datetime

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

original_df = pd.read_csv(CONFIG['data_path'])

df = original_df[(original_df["model"] == CONFIG['text_model']) | (original_df["model"] == "authentic")]

# Variables

output_file = f"{CONFIG['output_dir']}k_fold_author_validation_data.csv"

authors = df['author'].unique().tolist()

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

# Looping through authors

for author in authors:
    print(f"Running tests for {author}")

    # Create dataframes for the test and train
    one_author_df = df[df['author'] == author]
    nine_author_df = df[df['author'] != author]

    # 100 runs per author

    for i in range(100):
        print(f"Prepping data for test {i} for {author}")
        
        # Sample 100 rows for each of the other authors for the train set 
        train_sampled_rows = []
     
        for author in nine_author_df['author'].unique():
            x = int(100 / 2) 
            authentic_rows = nine_author_df[(nine_author_df['category'] == 'authentic') & (nine_author_df['author'] == author)]
            synthetic_rows = nine_author_df[(nine_author_df['category'] == 'synthetic') & (nine_author_df['author'] == author)]

            authentic_sample = authentic_rows.sample(n=x)
            synthetic_sample = synthetic_rows.sample(n=x)
            train_sampled_rows.extend([authentic_sample, synthetic_sample])

        # Combine all sampled rows into a single dataframe
        train_df = pd.concat(train_sampled_rows).reset_index(drop=True)

        test_sampled_rows = []

        for author in one_author_df['author'].unique():
            x = int(225 // 2) 
            authentic_rows = one_author_df[(one_author_df['category'] == 'authentic') & (one_author_df['author'] == author)]
            synthetic_rows = one_author_df[(one_author_df['category'] == 'synthetic') & (one_author_df['author'] == author)]
            authentic_sample = authentic_rows.sample(n=x)
            synthetic_sample = synthetic_rows.sample(n=x)
            test_sampled_rows.extend([authentic_sample, synthetic_sample])

        test_df = pd.concat(test_sampled_rows).reset_index(drop=True)

        # Combine train and test to get all sample IDs
        all_samples_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        all_sample_ids = all_samples_df['id'].tolist()
        train_ids = train_df['id'].tolist()
        test_ids = test_df['id'].tolist()


        # Feature and target extraction
        X_train = train_df[feature_cols]
        y_train = train_df['category']
        X_test = test_df[feature_cols]
        y_test = test_df['category']

        # Ensure X is defined properly for feature importance calculation
        X = pd.concat([X_train, X_test], ignore_index=True)

        print(f"Training model for test {i} for {author}")

        # Initialize the Random Forest model
        model = RandomForestClassifier(n_estimators=CONFIG['n_estimators'], random_state=CONFIG['random_state'])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        print(f"Getting evaluation data for test {i} for {author}")
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        
        print(accuracy)
        
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Calculate metrics for each category
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=['authentic', 'synthetic'])
    
        # Add the predicted labels to the test set for comparison
        X_test = X_test.copy()  
        X_test['actual_label'] = y_test
        X_test['predicted_label'] = y_pred
        
        # Join with the original dataframe to include `id` and `author` columns
        misclassified = X_test[X_test['actual_label'] != X_test['predicted_label']]
        misclassified = misclassified.join(test_df[['id', 'author']], how='left')
        
        # Display the misclassified rows with the required columns
        misclassified_result = misclassified[['id', 'author', 'actual_label', 'predicted_label']].sort_values(by=['author', 'actual_label'])

        # Create lists of chunk ids for misclassified authentic and synthetic
        authentic_mislabeled =  misclassified_result[ misclassified_result['actual_label'] == 'authentic']['id'].tolist()
        synthetic_mislabeled =  misclassified_result[ misclassified_result['actual_label'] == 'synthetic']['id'].tolist()
        
        # Create dataframe to store feature importance scores
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        
        # Sort features by importance
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)
   
        top_10_df = feature_importance[:10]
        
        top_10_features = dict(zip(top_10_df['feature'], top_10_df['importance']))
        
        print(f"Test {i} for {author} complete. Printing to {output_file}")
        

        # Printing results to csv
        
        author_holdout_test = pd.DataFrame(columns=['run_date',
                                            'test_author', 
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
        
        author_holdout_test.at[0, 'run_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        author_holdout_test.at[0, 'test_author'] = author
        author_holdout_test.at[0, 'features_used'] = feature_cols
        author_holdout_test.at[0, 'accuracy'] = accuracy
        
        author_holdout_test.at[0, 'Overall F1'] = overall_f1
        author_holdout_test.at[0, 'Overall Precision'] = overall_precision
        author_holdout_test.at[0, 'Overall Recall'] = overall_recall
        
        author_holdout_test.at[0, 'Authentic F1'] = round(f1[0], 5)
        author_holdout_test.at[0, 'Authentic Precision'] = round(precision[0], 5)
        author_holdout_test.at[0, 'Authentic Recall'] = round(recall[0], 5)
        
        author_holdout_test.at[0, 'Synthetic F1'] = round(f1[1], 5)
        author_holdout_test.at[0, 'Synthetic Precision'] = round(precision[1], 5)
        author_holdout_test.at[0, 'Synthetic Recall'] = round(recall[1], 5)
        
        author_holdout_test.at[0, 'authentic_mislabeled'] = authentic_mislabeled
        author_holdout_test.at[0, 'synthetic_mislabeled'] = synthetic_mislabeled
        author_holdout_test.at[0, 'top_10_features'] = top_10_features
        author_holdout_test.at[0, 'all_sample_ids'] = all_sample_ids
        author_holdout_test.at[0, 'train_ids'] = train_ids
        author_holdout_test.at[0, 'test_ids'] = test_ids
        
        
        if os.path.exists(output_file):
            author_holdout_test.to_csv(output_file, mode='a', header=False, index=False)
        else:
            author_holdout_test.to_csv(output_file, index=False)
        
        print(f"Results added to {output_file}")
    
    
    


