import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
        # random_state=42: Ensures reproducible results.
        # max_iter=1000: Sets the maximum number of iterations for the solver to converge.
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Linear SVC (SVM)': LinearSVC(random_state=42)
    }

    targets = ['project_name', 'Type', 'Priority']
    X = train_df['text']

    print("\n#### Starting Model Evaluation (5-Fold Cross-Validation)")

    for target in targets:
        print(f"\nEvaluating models for target: '{target}'")
        y = train_df[target]

        for model_name, classifier in models.items():
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', classifier)
            ])

            # Perform cross-validation and calculate mean and std of accuracy
            # pipeline: The model to evaluate.
            # X: The feature data.
            # y: The target labels.
            # cv=5: Use 5-fold cross-validation.
            # scoring='accuracy': Evaluate performance using accuracy.
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  - {model_name:<25}: Mean Accuracy = {mean_score:.3f} (+/- {std_score:.3f})")
            
    print("\n#### Model Evaluation Complete")

def train_final_models_and_predict(train_df, predict_df):
    # stop_words='english': Removes common English stop words.
    # min_df=2: Ignores words that appear in less than 2 documents.
    # ngram_range=(1, 2): Considers both single words (unigrams) and word pairs (bigrams).
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))
    X_train = tfidf_vectorizer.fit_transform(train_df['text'])
    X_predict = tfidf_vectorizer.transform(predict_df['text'])

    y_trains = {
        'project_name': train_df['project_name'],
        'Type': train_df['Type'],
        'Priority': train_df['Priority']
    }

    best_models = {
        'project_name': LinearSVC(random_state=42),
        # random_state=42: Ensures reproducible results.
        # max_iter=1000: Sets the maximum number of iterations for the solver to converge.
        'Type': LogisticRegression(random_state=42, max_iter=1000),
        'Priority': LogisticRegression(random_state=42, max_iter=1000)
    }

    processed_df = predict_df.copy()

    for target_name, model in best_models.items():
        print(f"Training and predicting for '{target_name}'...")
        model.fit(X_train, y_trains[target_name])
        predictions = model.predict(X_predict)
        processed_df[target_name] = predictions
        
    print("Prediction complete.\n")
    return processed_df

def evaluate_predictions(predicted_df, labels_file_path):
    print("\n#### Evaluating Predictions Against Manual Labels")
    
    try:
        manual_labels_df = pd.read_json(labels_file_path)
    except FileNotFoundError:
        print(f"Warning: Manual labels file not found at '{labels_file_path}'. Skipping evaluation.")
        return

    eval_df = pd.merge(predicted_df, manual_labels_df, on='id', suffixes=('_pred', '_true'))
    targets = ['project_name', 'Type', 'Priority']

    for target in targets:
        true_col = f'{target}_true'
        pred_col = f'{target}_pred'

        print(f"===== Evaluation Report for: {target} =====")
        report = classification_report(eval_df[true_col], eval_df[pred_col], zero_division=0)
        print(report)

        # eval_df[true_col]: The actual true labels.
        # eval_df[pred_col]: The predicted labels.
        cm = confusion_matrix(eval_df[true_col], eval_df[pred_col], labels=eval_df[true_col].unique())
        plt.figure(figsize=(10, 7))
        # annot=True: Displays the numbers in each cell.
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=eval_df[true_col].unique(), 
                    yticklabels=eval_df[true_col].unique())
        plt.title(f'Confusion Matrix for {target}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'images/confusion_matrix_{target.lower()}.png', bbox_inches='tight')
        plt.close()

def main():
    DATA_FILE_PATH = 'data/dataset.json'
    MANUAL_LABELS_PATH = 'data/manual_labels_dataset.json'
    OUTPUT_PATH = 'data/processed_dataset.json'

    train_df, predict_df = load_data(DATA_FILE_PATH)
    train_evaluate_models(train_df)

    final_df = train_final_models_and_predict(train_df, predict_df)

    evaluate_predictions(final_df, MANUAL_LABELS_PATH)

    final_df_to_save = final_df.drop(columns=['text'], errors='ignore')
    output_json_data = final_df_to_save.to_dict(orient='records')
    with open(OUTPUT_PATH, 'w') as f:
        import json
        json.dump(output_json_data, f, indent=2)

if __name__ == "__main__":
    main()