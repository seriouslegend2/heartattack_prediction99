# Heart Attack Analysis and Prediction Project

## Project Overview

This project is focused on the analysis and prediction of heart attack risks based on medical attributes such as age, blood pressure, cholesterol, maximum heart rate, and others. Using machine learning techniques, the goal is to predict whether a patient is likely to suffer from a heart attack, which is a critical health problem worldwide.

### Problem Statement

Heart disease is one of the leading causes of death globally. Early detection and prediction of heart disease can help healthcare professionals make better decisions about patient treatment. The primary goal of this project is to build a machine learning model that can predict whether a patient is at risk of a heart attack, given a set of health-related features.

### Dataset

The dataset used in this project is the **Heart Attack Prediction Dataset** available on [Kaggle](https://www.kaggle.com/). The dataset contains 303 instances and 14 features, including both numerical and categorical data. The target variable, `target`, indicates whether a patient has a heart attack (1) or not (0).

#### Features in the Dataset:
1. **Age**: Age of the patient
2. **Sex**: Sex of the patient (1 = male; 0 = female)
3. **Cp**: Chest pain type (4 possible values)
4. **Trtbps**: Resting blood pressure (in mm Hg)
5. **Chol**: Serum cholesterol (in mg/dl)
6. **Fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **Restecg**: Resting electrocardiographic results (3 possible values)
8. **Thalach**: Maximum heart rate achieved
9. **Exang**: Exercise induced angina (1 = yes; 0 = no)
10. **Oldpeak**: ST depression induced by exercise relative to rest
11. **Slope**: The slope of the peak exercise ST segment (3 possible values)
12. **Ca**: Number of major vessels colored by fluoroscopy (0-3)
13. **Thal**: 3 possible values related to thalassemia
14. **Target**: Target variable (1 = heart attack risk; 0 = no heart attack risk)

### Objective

The primary objective of this project is to predict the target variable (`target`), which indicates whether the patient is at risk of a heart attack. By using machine learning algorithms, we will predict the likelihood of a heart attack based on the patient's health data. The models will be evaluated based on their performance using metrics like accuracy, precision, recall, and F1-score.

### Project Phases

The project is divided into several key phases:

#### 1. Data Loading and Exploration

The first step involved loading the dataset and examining its structure. This phase includes checking for missing values, understanding the types of data in each column, and renaming columns to ensure clarity.

#### 2. Exploratory Data Analysis (EDA)

After the initial data inspection, the next step is to explore the dataset further. EDA includes visualizing the distribution of variables, analyzing correlations between features, and understanding patterns that may influence the model's performance. Key techniques include:
- Visualizing numerical features using histograms, box plots, and scatter plots.
- Analyzing correlations between features using a correlation heatmap.
- Identifying trends or patterns in the dataset that could be important for prediction.

#### 3. Data Preprocessing

The raw data was preprocessed to make it suitable for machine learning algorithms. This step includes:
- **Handling missing values**: The dataset did not contain any missing values, so no imputation was necessary.
- **Feature scaling**: To bring all numerical features to a common scale, standardization was applied using `StandardScaler`.
- **Encoding categorical variables**: Categorical variables were encoded as numerical values (if necessary), which is important for machine learning algorithms that require numerical input.

#### 4. Model Building

In this phase, multiple machine learning models were applied to the preprocessed dataset to predict the likelihood of a heart attack. The models used in this project include:
- **Logistic Regression**: A statistical model used for binary classification.
- **Decision Trees**: A tree-based algorithm used for both classification and regression tasks.
- **Random Forests**: An ensemble method that combines multiple decision trees to improve accuracy.

#### 5. Model Evaluation

Once the models were trained, they were evaluated using a test set (20% of the data). Performance metrics such as accuracy, precision, recall, and F1-score were used to assess the effectiveness of the models. The best-performing model was selected based on these metrics.

#### 6. Hyperparameter Tuning (Optional)

For some models (such as Decision Trees and Random Forests), hyperparameter tuning was performed using techniques like GridSearchCV to find the best set of hyperparameters for improved model performance.

### Results

After training and evaluating the models, the **Random Forest classifier** emerged as the best-performing model with the highest accuracy in predicting heart attack risk. The final model was able to make accurate predictions based on the available features, making it a suitable candidate for deployment in real-world healthcare settings.

### Conclusion

This project demonstrates the power of machine learning in predicting heart disease risk based on health-related data. Through the use of various classification algorithms and data analysis techniques, we were able to build a reliable model that can assist healthcare professionals in early detection and intervention. The results from this project highlight the potential of using machine learning to support decision-making in the healthcare industry.

### Future Work

- **Data augmentation**: If more data becomes available, augmenting the dataset could improve the model's generalization.
- **Model interpretability**: Further work could involve interpreting the model's decision-making process using techniques like SHAP or LIME to better understand the influence of each feature.
- **Integration into healthcare systems**: The final model could be integrated into healthcare decision-support systems to assist doctors in diagnosing heart conditions.

### Libraries Used

The following Python libraries were used throughout this project:

- `pandas` for data manipulation and analysis
- `numpy` for numerical computations
- `matplotlib` and `seaborn` for data visualization
- `scikit-learn` for machine learning algorithms and model evaluation

---

## How to Run the Code

To run the project on your local machine or in a cloud environment, you will need to install the required libraries listed above. After that, you can simply load the dataset and execute the code in the provided Jupyter Notebooks or Python scripts.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
cvbnfxn
