# Issue Classifier

This repository contains the solution for the programming task.

## Exploratory Data Analysis (EDA)

Before building any models, I performed a thorough exploratory analysis on the first 50 issues (the training set) to understand the data's structure, distributions, and content. This process ensures that all subsequent modeling decisions are data-driven.

### 1. Data Snapshot

A quick snapshot of the data reveals the fields we will be working with. The primary features for our model will be the `summary` and `description` text.

![Data Snapshot](images/data_snapshot.png)

### 2. Target Variable Distributions

An analysis of the three target variables (`project_name`, `Type`, `Priority`) revealed clear class imbalances.

![Target Variable Distributions](images/target_distributions.png)

* **Note:** The `project_name` and `Priority` fields are heavily skewed. As the chart shows, `Unidentified Roe` is the most frequent project, and `Normal` is the dominant priority. This imbalance must be considered, as a naive model might simply over-predict the majority class. The `Type` field is also imbalanced, with `Bug` and `Task` being the most common types.

### 3. Text Feature Characteristics

The lengths of the text fields were analyzed to understand the data's complexity and information density.

![Text Length Distribution](images/text_length_distribution.png)

* **Note:** While `summary` lengths are relatively consistent (mostly between 60-100 characters), `description` lengths vary widely, with a great portion containing over 600 characters. This suggests that the description holds rich, detailed information that will be vital for the model to distinguish between different issue types and projects. This justifies combining both fields for feature extraction.

### 4. Keyword and Content Analysis

To understand the content itself, I generated word clouds. Comparing the overall keywords with those from the two most frequent projects (`Unidentified Roe` and `Fast Badger`) yielded powerful insights.

![Word Cloud Comparison](images/wordcloud_comparison.png)

* **Note:** The keyword analysis provides strong evidence that text content is a viable predictor. The "Overall" cloud shows common terms like `issue`, `problem`, and `details`. However, the project-specific clouds reveal distinct vocabularies. `Unidentified Roe` issues frequently mention UI/UX related terms like `icon`, `menu`, `color`, and `duplicate`, while `Fast Badger` issues are dominated by system-level terms like `plugin`, `system`, and `notification`. This clear separation in terminology gives confidence that a machine learning model can effectively learn and leverage these patterns.

---

## Model Selection

To select the most suitable algorithm for this task, I evaluated three baseline models for text classification. This process was performed using 5-fold cross-validation on the training set to ensure the performance estimates are stable and reliable.

### 1. Choice of Models

I chose the following three models to establish a baseline:
* **Logistic Regression:** A linear model that serves as a strong baseline.
* **Multinomial Naive Bayes:** A classic probabilistic model that is traditionally very effective for text classification.
* **Linear SVC (SVM):** A model based on geometric principles, which has high performance on high-dimensional, sparse data like TF-IDF features.

### 2. Cross-Validation Results

The models were evaluated on their mean accuracy across the 5 folds. The results are summarized below:

| Target         | Logistic Regression | Multinomial Naive Bayes | Linear SVC (SVM)        |
| :------------- | :------------------ | :---------------------- | :---------------------- |
| `project_name` | 0.180 (+/- 0.040)   | 0.180 (+/- 0.040)       | **0.320 (+/- 0.204)** |
| `Type`         | **0.600 (+/- 0.000)** | 0.600 (+/- 0.000)       | 0.600 (+/- 0.000)       |
| `Priority`     | **0.760 (+/- 0.049)** | 0.760 (+/- 0.049)       | 0.760 (+/- 0.049)       |

### 3. Analysis and Decision

* **Note**: The evaluation triggered a `UserWarning` means that all three target variables have classes with only one member. This confirms the severe class imbalance discovered during the EDA and explains the challenges in achieving high accuracy.

* **For `project_name`**: The scores are low across the board, which is expected given the extreme class imbalance. However, **Linear SVC** shows a notably higher, although unstable, performance. It is selected as the best available option for this challenging target.

* **For `Priority`**: Critically, the high accuracy of 76% is misleading. My Above EDA shows that the majority class (`Normal`) constitutes approximately 76% of the data. This score shows that the models are likely not learning complex patterns but are defaulting to predicting the majority class.

* **Decision**: Based on these results, the following models are chosen for the final prediction task:
    * **`project_name`**: **Linear SVC (SVM)**
    * **`Type`**: **Logistic Regression** (chosen for its simplicity as all models performed identically)
    * **`Priority`**: **Logistic Regression** (chosen for simplicity, while acknowledging its performance is baseline)