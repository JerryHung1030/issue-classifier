import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Load the data and split it into training(50) and prediction(50) sets
def load_data(file_path):
    print("Loading data...")
    df = pd.read_json(file_path)

    train_df = df.iloc[:50].copy()
    predict_df = df.iloc[50:].copy()

    # Combine 'summary' and 'description' into a single text column
    train_df['text'] = train_df['summary'] + ' ' + train_df['description']
    predict_df['text'] = predict_df['summary'] + ' ' + predict_df['description']
    
    print(f"Data loaded: {len(train_df)} training samples, {len(predict_df)} prediction samples.")
    return train_df, predict_df

# Use 5-Fold Cross-Validation to evaluate multiple models on different variables
def train_evaluate_models(train_df):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Linear SVC (SVM)': LinearSVC(random_state=42)
    }

    targets = ['project_name', 'Type', 'Priority']
    X = train_df['text']

    print("\n--- Starting Model Evaluation (5-Fold Cross-Validation) ---")

    for target in targets:
        print(f"\nEvaluating models for target: '{target}'")
        y = train_df[target]

        for model_name, classifier in models.items():
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', classifier)
            ])

            # Perform cross-validation and calculate mean and std of accuracy
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  - {model_name:<25}: Mean Accuracy = {mean_score:.3f} (+/- {std_score:.3f})")
            
    print("\n--- Model Evaluation Complete ---")

# Main function to load data and evaluate models
def main():
    DATA_FILE_PATH = 'data/dataset.json'
    train_df, _ = load_data(DATA_FILE_PATH)
    train_evaluate_models(train_df)

if __name__ == "__main__":
    main()