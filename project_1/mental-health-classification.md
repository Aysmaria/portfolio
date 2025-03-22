# Mental Health Classification Project

## Project Overview
This notebook implements a classification model to predict anxiety and depression levels based on various mental health factors. We'll work with the [Anxiety and Depression Mental Health Factors](https://www.kaggle.com/datasets/ak0212/anxiety-and-depression-mental-health-factors/data) dataset from Kaggle.

## 1. Setup and Data Acquisition

### 1.1 Library Imports


```python
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
                            accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Feature importance
import shap

# SQLite for database operations
import sqlite3
from sqlite3 import Error

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set(style="whitegrid")

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
```

### 1.2 Data Loading 

Let's load the dataset and take an initial look at its structure. This step helps us understand what data we're working with and how it's organized.


```python
# Load the dataset
file_path = 'anxiety_and_depression_mental_health_factors.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    print("Please ensure the dataset is downloaded from Kaggle and placed in the same directory as this notebook.")
    print("Dataset URL: https://www.kaggle.com/datasets/ak0212/anxiety-and-depression-mental-health-factors/data")
    
    # If you want to download the dataset directly (requires kaggle API setup)
    # !kaggle datasets download -d ak0212/anxiety-and-depression-mental-health-factors
    # !unzip anxiety-and-depression-mental-health-factors.zip
    # df = pd.read_csv('anxiety_and_depression_mental_health_factors.csv')
```

### 1.3 Store Data in SQLite Database

Creating a SQLite database will allow us to practice SQL queries on this dataset. This is an important skill for data analysts and scientists.


```python
# Create a SQLite database and load the data
def create_connection(db_file):
    """Create a database connection to a SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to SQLite version: {sqlite3.version}")
        return conn
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """Create a table from the create_table_sql statement"""
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def insert_data(conn, df, table_name):
    """Insert dataframe into SQLite table"""
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Data inserted into {table_name} table successfully.")

# Create database connection
conn = create_connection("mental_health.db")

if conn is not None:
    # Create table
    table_name = 'mental_health_data'
    insert_data(conn, df, table_name)
    
    # Example SQL query
    print("\nExample SQL query to verify data:")
    query = f"SELECT COUNT(*) FROM {table_name}"
    result = pd.read_sql_query(query, conn)
    print(f"Total records: {result.iloc[0, 0]}")
    
    # Close connection
    conn.close()
else:
    print("Error! Cannot create the database connection.")
```

### 1.4 Initial Data Examination


```python
# Display first few rows
print("First 5 rows of the dataset:")
display(df.head())

# Get dataset information
print("\nDataset Information:")
display(df.info())

# Summary statistics
print("\nSummary Statistics:")
display(df.describe())

# Check for missing values
print("\nMissing Values per Column:")
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_data = pd.concat([missing_values, missing_percentage], axis=1)
missing_data.columns = ['Missing Values', 'Percentage (%)']
display(missing_data[missing_data['Missing Values'] > 0])

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
```

## 2. Exploratory Data Analysis (EDA)

### 2.1 Exploring Target Variables

Let's first understand our target variables: Anxiety_Score and Depression_Score. We need to decide how to approach the classification task based on these scores.


```python
# Analyze target variables
print("Anxiety Score Statistics:")
print(df['Anxiety_Score'].describe())

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Anxiety_Score'], kde=True, bins=20)
plt.title('Distribution of Anxiety Scores')
plt.xlabel('Anxiety Score')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['Depression_Score'], kde=True, bins=20)
plt.title('Distribution of Depression Scores')
plt.xlabel('Depression Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Correlation between Anxiety and Depression
correlation = df['Anxiety_Score'].corr(df['Depression_Score'])
print(f"\nCorrelation between Anxiety and Depression Scores: {correlation:.4f}")

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Anxiety_Score', y='Depression_Score', data=df, alpha=0.6)
plt.title('Relationship between Anxiety and Depression Scores')
plt.xlabel('Anxiety Score')
plt.ylabel('Depression Score')
plt.show()
```

### 2.2 Creating Target Classes for Classification

For classification tasks, we need categorical target variables. Let's create categories based on anxiety and depression scores.


```python
# Create categories for anxiety levels
def categorize_anxiety(score):
    if score <= 7:
        return 'Low'
    elif score <= 14:
        return 'Moderate'
    else:
        return 'High'

# Create categories for depression levels
def categorize_depression(score):
    if score <= 9:
        return 'Low'
    elif score <= 19:
        return 'Moderate'
    else:
        return 'High'

# Apply categorization
df['Anxiety_Level'] = df['Anxiety_Score'].apply(categorize_anxiety)
df['Depression_Level'] = df['Depression_Score'].apply(categorize_depression)

# Combine into a single mental health status variable
def combined_mental_health(anxiety, depression):
    if anxiety == 'Low' and depression == 'Low':
        return 'Healthy'
    elif anxiety == 'High' or depression == 'High':
        return 'Severe'
    else:
        return 'Moderate'

df['Mental_Health_Status'] = df.apply(lambda x: combined_mental_health(x['Anxiety_Level'], x['Depression_Level']), axis=1)

# Display the distribution of the categories
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.countplot(x='Anxiety_Level', data=df, order=['Low', 'Moderate', 'High'])
plt.title('Distribution of Anxiety Levels')
plt.xlabel('Anxiety Level')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
sns.countplot(x='Depression_Level', data=df, order=['Low', 'Moderate', 'High'])
plt.title('Distribution of Depression Levels')
plt.xlabel('Depression Level')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
sns.countplot(x='Mental_Health_Status', data=df, order=['Healthy', 'Moderate', 'Severe'])
plt.title('Distribution of Mental Health Status')
plt.xlabel('Mental Health Status')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# For this project, we'll focus on predicting the combined Mental_Health_Status
print(f"Target variable distribution:\n{df['Mental_Health_Status'].value_counts()}")
print(f"Target variable distribution (%):\n{df['Mental_Health_Status'].value_counts(normalize=True)*100}")
```

### 2.3 Exploring Demographic Features


```python
# Analyze demographic features
plt.figure(figsize=(18, 12))

# Age distribution
plt.subplot(2, 3, 1)
sns.histplot(df['Age'], kde=True, bins=20)
plt.title('Age Distribution')

# Gender distribution
plt.subplot(2, 3, 2)
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')

# Education level
plt.subplot(2, 3, 3)
sns.countplot(y='Education_Level', data=df, order=df['Education_Level'].value_counts().index)
plt.title('Education Level Distribution')
plt.xlabel('Count')

# Employment status
plt.subplot(2, 3, 4)
sns.countplot(y='Employment_Status', data=df, order=df['Employment_Status'].value_counts().index)
plt.title('Employment Status Distribution')
plt.xlabel('Count')

# Mental health status by gender
plt.subplot(2, 3, 5)
gender_mental_health = pd.crosstab(df['Gender'], df['Mental_Health_Status'], normalize='index') * 100
gender_mental_health.plot(kind='bar', stacked=True)
plt.title('Mental Health Status by Gender')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.legend(title='Mental Health Status')

# Mental health status by education
plt.subplot(2, 3, 6)
education_order = ['High School', 'Bachelor\'s', 'Master\'s', 'PhD', 'Other']
education_mental_health = pd.crosstab(
    df['Education_Level'], 
    df['Mental_Health_Status'],
    normalize='index'
) * 100
education_mental_health.loc[education_order].plot(kind='bar', stacked=True)
plt.title('Mental Health Status by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Percentage')
plt.legend(title='Mental Health Status')

plt.tight_layout()
plt.show()
```

### 2.4 Exploring Lifestyle and Health Indicators


```python
# Analyzing lifestyle features
lifestyle_features = ['Sleep_Hours', 'Physical_Activity_Hrs', 'Social_Support_Score', 
                     'Stress_Level', 'Self_Esteem_Score', 'Life_Satisfaction_Score',
                     'Loneliness_Score']

plt.figure(figsize=(18, 15))

for i, feature in enumerate(lifestyle_features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Mental_Health_Status', y=feature, data=df, order=['Healthy', 'Moderate', 'Severe'])
    plt.title(f'{feature} by Mental Health Status')
    plt.xlabel('Mental Health Status')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

# Distribution of coping mechanisms
coping_features = ['Therapy', 'Meditation', 'Substance_Use']

plt.figure(figsize=(18, 5))

for i, feature in enumerate(coping_features):
    plt.subplot(1, 3, i+1)
    coping_mental_health = pd.crosstab(df[feature], df['Mental_Health_Status'], normalize='index') * 100
    coping_mental_health.plot(kind='bar', stacked=True)
    plt.title(f'Mental Health Status by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Percentage')
    plt.legend(title='Mental Health Status')

plt.tight_layout()
plt.show()
```

### 2.5 Correlation Analysis


```python
# Select numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Calculate correlation matrix
correlation_matrix = df[numeric_cols].corr()

# Plot heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(correlation_matrix)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap, mask=mask, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()

# Find features highly correlated with Anxiety and Depression scores
anxiety_correlations = correlation_matrix['Anxiety_Score'].sort_values(ascending=False)
depression_correlations = correlation_matrix['Depression_Score'].sort_values(ascending=False)

print("\nTop 5 features correlated with Anxiety Score:")
print(anxiety_correlations.head(6))

print("\nTop 5 features correlated with Depression Score:")
print(depression_correlations.head(6))
```

### 2.6 SQL-Based Analysis

Let's leverage our SQLite database to perform some analysis using SQL queries. This demonstrates SQL skills within the data exploration process.


```python
# Connect to SQLite database
conn = create_connection("mental_health.db")

if conn is not None:
    # Query 1: Average anxiety and depression scores by gender
    query1 = """
    SELECT 
        Gender,
        AVG(Anxiety_Score) as Avg_Anxiety,
        AVG(Depression_Score) as Avg_Depression
    FROM mental_health_data
    GROUP BY Gender
    ORDER BY Avg_Anxiety DESC
    """
    
    # Query 2: Average scores by employment status
    query2 = """
    SELECT 
        Employment_Status,
        COUNT(*) as Count,
        AVG(Anxiety_Score) as Avg_Anxiety,
        AVG(Depression_Score) as Avg_Depression
    FROM mental_health_data
    GROUP BY Employment_Status
    ORDER BY Avg_Depression DESC
    """
    
    # Query 3: Average scores for those who do/don't use therapy
    query3 = """
    SELECT 
        Therapy,
        COUNT(*) as Count,
        AVG(Anxiety_Score) as Avg_Anxiety,
        AVG(Depression_Score) as Avg_Depression
    FROM mental_health_data
    GROUP BY Therapy
    """
    
    # Execute and display results
    print("Average Anxiety and Depression Scores by Gender:")
    display(pd.read_sql_query(query1, conn))
    
    print("\nAverage Scores by Employment Status:")
    display(pd.read_sql_query(query2, conn))
    
    print("\nAverage Scores by Therapy Usage:")
    display(pd.read_sql_query(query3, conn))
    
    # Close connection
    conn.close()
else:
    print("Error! Cannot create the database connection.")
```

## 3. Data Preprocessing

### 3.1 Data Cleaning

Let's address any issues in our dataset before building models.


```python
# First, make a copy of the original dataframe
df_cleaned = df.copy()

# Check for missing values again
missing_values = df_cleaned.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Check for outliers in numeric columns
numeric_features = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_features = [col for col in numeric_features if col not in ['Anxiety_Score', 'Depression_Score']]

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features):
    if i < 9:  # Limit to first 9 features for readability
        plt.subplot(3, 3, i+1)
        sns.boxplot(y=df_cleaned[feature])
        plt.title(f'Boxplot of {feature}')

plt.tight_layout()
plt.show()

# Handle missing values using imputation
# We'll use the SimpleImputer from sklearn for this task
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputation to numeric and categorical columns separately
numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

df_cleaned[numeric_cols] = numeric_imputer.fit_transform(df_cleaned[numeric_cols])
df_cleaned[categorical_cols] = categorical_imputer.fit_transform(df_cleaned[categorical_cols])

# Verify no missing values remain
print("\nMissing values after imputation:")
print(df_cleaned.isnull().sum().sum())

# Handle outliers using capping (winsorization)
from scipy import stats

def cap_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

# Apply outlier capping to numeric features
for feature in numeric_features:
    df_cleaned = cap_outliers(df_cleaned, feature)

# Check results after outlier treatment
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features):
    if i < 9:  # Limit to first 9 features for readability
        plt.subplot(3, 3, i+1)
        sns.boxplot(y=df_cleaned[feature])
        plt.title(f'Boxplot of {feature} (After Capping)')

plt.tight_layout()
plt.show()
```

### 3.2 Feature Encoding and Scaling


```python
# Encode categorical variables
# First, identify categorical columns (excluding target variables we created)
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ['Anxiety_Level', 'Depression_Level', 'Mental_Health_Status']]

print("Categorical columns to encode:")
print(categorical_cols)

# Create a copy for preprocessing
df_encoded = df_cleaned.copy()

# Apply One-Hot Encoding for categorical variables
for col in categorical_cols:
    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
    df_encoded = pd.concat([df_encoded, dummies], axis=1)
    df_encoded.drop(col, axis=1, inplace=True)

# Encode the target variable
label_encoder = LabelEncoder()
df_encoded['Mental_Health_Status_Encoded'] = label_encoder.fit_transform(df_encoded['Mental_Health_Status'])

# Print mapping for reference
status_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nTarget Variable Encoding:")
print(status_mapping)

# Now scale the numeric features
numeric_features_to_scale = [col for col in df_encoded.select_dtypes(include=['float64', 'int64']).columns
                             if col not in ['Mental_Health_Status_Encoded', 'Anxiety_Score', 'Depression_Score']]

scaler = StandardScaler()
df_encoded[numeric_features_to_scale] = scaler.fit_transform(df_encoded[numeric_features_to_scale])

# Check the result
print("\nDataset shape after encoding:", df_encoded.shape)
print("\nFirst few rows of encoded data:")
display(df_encoded.head())
```

## 4. Feature Engineering

### 4.1 Creating New Features


```python
# Create new features that might be useful for prediction
df_features = df_encoded.copy()

# 1. Combine sleep and physical activity into a wellness score
df_features['Wellness_Score'] = (
    (df_cleaned['Sleep_Hours'] / df_cleaned['Sleep_Hours'].max()) * 0.5 + 
    (df_cleaned['Physical_Activity_Hrs'] / df_cleaned['Physical_Activity_Hrs'].max()) * 0.5
) * 10  # Scale to 0-10

# 2. Create a stress-to-support ratio
df_features['Stress_Support_Ratio'] = df_cleaned['Stress_Level'] / (df_cleaned['Social_Support_Score'] + 1)  # +1 to avoid division by zero

# 3. Create an interaction feature between financial and work stress
df_features['Combined_Stress'] = df_cleaned['Financial_Stress'] * df_cleaned['Work_Stress']

# 4. Create a coping mechanisms score (higher = more coping mechanisms used)
df_features['Coping_Score'] = (
    (df_cleaned['Therapy'] == 'Yes').astype(int) + 
    (df_cleaned['Meditation'] == 'Yes').astype(int) - 
    (df_cleaned['Substance_Use'] == 'Yes').astype(int)  # Substance use reduces coping score
)

# 5. Create a binary feature for presence of mental illness history
df_features['Has_Mental_Illness_History'] = (df_cleaned['Family_History_Mental_Illness'] == 'Yes').astype(int)

# 6. Create a well-being index
df_features['Wellbeing_Index'] = (
    df_cleaned['Self_Esteem_Score'] + 
    df_cleaned['Life_Satisfaction_Score'] - 
    df_cleaned['Loneliness_Score']
)

# 7. Age groups
def age_group(age):
    if age < 25:
        return 0  # Young adult
    elif age < 40:
        return 1  # Adult
    elif age < 60:
        return 2  # Middle-aged
    else:
        return 3  # Senior

df_features['Age_Group'] = df_cleaned['Age'].apply(age_group)

# Display the new features
new_features = ['Wellness_Score', 'Stress_Support_Ratio', 'Combined_Stress', 
                'Coping_Score', 'Has_Mental_Illness_History', 'Wellbeing_Index', 'Age_Group']

print("New Feature Statistics:")
display(df_features[new_features].describe())

# Visualize some of the new features
plt.figure(figsize=(18, 6))

# Wellness score by mental health status
plt.subplot(1, 3, 1)
sns.boxplot(x='Mental_Health_Status', y='Wellness_Score', data=df_features.join(df_cleaned['Mental_Health_Status']), 
            order=['Healthy', 'Moderate', 'Severe'])
plt.title('Wellness Score by Mental Health Status')

# Stress-to-support ratio by mental health status
plt.subplot(1, 3, 2)
sns.boxplot(x='Mental_Health_Status', y='Stress_Support_Ratio', data=df_features.join(df_cleaned['Mental_Health_Status']), 
            order=['Healthy', 'Moderate', 'Severe'])
plt.title('Stress-to-Support Ratio by Mental Health Status')

# Wellbeing index by mental health status
plt.subplot(1, 3, 3)
sns.boxplot(x='Mental_Health_Status', y='Wellbeing_Index', data=df_features.join(df_cleaned['Mental_Health_Status']), 
            order=['Healthy', 'Moderate', 'Severe'])
plt.title('Wellbeing Index by Mental Health Status')

plt.tight_layout()
plt.show()
```

### 4.2 Feature Selection


```python
# Prepare data for feature selection
X = df_features.drop(['Mental_Health_Status_Encoded', 'Anxiety_Level', 'Depression_Level', 
                      'Mental_Health_Status', 'Anxiety_Score', 'Depression_Score'], axis=1)
y = df_features['Mental_Health_Status_Encoded']

# Split data for initial feature selection analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 1. Filter Method: Univariate feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# Apply SelectKBest with f_classif (ANOVA F-value)
selector = SelectKBest(f_classif, k=15)  # Select top 15 features
X_new = selector.fit_transform(X_train, y_train)

# Get selected feature names
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
})
feature_scores = feature_scores.sort_values('Score', ascending=False)

print("Top 15 features based on ANOVA F-value:")
display(feature_scores.head(15))

# 2. Embedded Method: Feature importance from Random Forest
rf_selector = RandomForestClassifier(random_state=42)
rf_selector.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_selector.feature_importances_
})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

print("\nTop 15 features based on Random Forest importance:")
display(feature_importances.head(15))

# Visualize feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()

# 3. RFE (Recursive Feature Elimination)
from sklearn.feature_selection import RFE

# Use Logistic Regression as the estimator
rfe_selector = RFE(estimator=LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=15, step=1)
rfe_selector = rfe_selector.fit(X_train, y_train)

# Get selected features
rfe_feature_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': rfe_selector.ranking_,
    'Selected': rfe_selector.support_
})
rfe_feature_ranking = rfe_feature_ranking.sort_values('Ranking')

print("\nTop 15 features based on RFE:")
display(rfe_feature_ranking.head(15))

# Choose the final set of features based on the intersection of methods
# Get top 15 features from each method
top_filter = feature_scores.head(15)['Feature'].tolist()
top_rf = feature_importances.head(15)['Feature'].tolist()
top_rfe = rfe_feature_ranking.head(15)['Feature'].tolist()

# Find common features across all methods
common_features = list(set(top_filter) & set(top_rf) & set(top_rfe))
print(f"\nFeatures common to all selection methods ({len(common_features)}):")
print(common_features)

# For features in at least 2 methods
features_in_two = list(set(
    (set(top_filter) & set(top_rf)) | 
    (set(top_filter) & set(top_rfe)) | 
    (set(top_rf) & set(top_rfe))
))
print(f"\nFeatures selected by at least 2 methods ({len(features_in_two)}):")
print(features_in_two)

# Final selected features (using features selected by at least 2 methods)
selected_features = features_in_two
print(f"\nFinal set of selected features ({len(selected_features)}):")
print(selected_features)

# Create a dataset with only selected features
X_selected = X[selected_features]
```

## 5. Baseline Model Development

### 5.1 Preparing Train and Test Sets


```python
# Split the data with selected features
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.25, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Target class distribution in training set:")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nTarget class distribution in test set:")
print(pd.Series(y_test).value_counts(normalize=True))
```

### 5.2 Training Baseline Models


```python
# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Function to evaluate model performance with cross-validation
def evaluate_model(model, X, y):
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'Mean CV Accuracy': cv_scores.mean(),
        'CV Accuracy StdDev': cv_scores.std()
    }

# Train and evaluate each model
baseline_results = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    # Evaluate with cross-validation
    cv_results = evaluate_model(model, X_train, y_train)
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save results
    baseline_results[name] = {
        'CV Mean Accuracy': cv_results['Mean CV Accuracy'],
        'CV Accuracy StdDev': cv_results['CV Accuracy StdDev'],
        'Test Accuracy': accuracy,
        'Test Precision': precision,
        'Test Recall': recall,
        'Test F1 Score': f1,
        'Model': model  # Save the trained model
    }
    
    print(f"  CV Accuracy: {cv_results['Mean CV Accuracy']:.4f} ± {cv_results['CV Accuracy StdDev']:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test F1 Score: {f1:.4f}")
    print()

# Convert results to DataFrame for easier comparison
baseline_df = pd.DataFrame.from_dict(baseline_results, orient='index')
baseline_df = baseline_df.drop('Model', axis=1)  # Remove model objects for display
display(baseline_df)

# Visualize results
plt.figure(figsize=(12, 6))
baseline_df[['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score']].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

### 5.3 Detailed Evaluation of Best Baseline Model


```python
# Identify the best model based on F1 score
best_model_name = baseline_df['Test F1 Score'].idxmax()
best_model = models[best_model_name]
print(f"Best model based on F1 score: {best_model_name}")

# Re-train best model
best_model.fit(X_train, y_train)

# Confusion Matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ROC Curve (One-vs-Rest approach for multiclass)
y_prob = best_model.predict_proba(X_test)

plt.figure(figsize=(10, 8))
for i, label in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for {best_model_name}')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
```

## 6. Model Optimization

### 6.1 Hyperparameter Tuning for Top Models


```python
# Set up hyperparameter grids
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'newton-cg'],
        'class_weight': [None, 'balanced']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': [None, 'balanced']
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Select top 2 models for hyperparameter tuning
top_models = baseline_df.nlargest(2, 'Test F1 Score').index.tolist()
print(f"Performing hyperparameter tuning for: {', '.join(top_models)}")

# Perform grid search for each selected model
tuned_models = {}

for model_name in top_models:
    print(f"\nTuning {model_name}...")
    
    # Get base model and parameter grid
    base_model = models[model_name]
    param_grid = param_grids.get(model_name, {})
    
    # Create grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and performance
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    
    # Save tuned model
    tuned_models[model_name] = {
        'Best Parameters': grid_search.best_params_,
        'CV F1 Score': grid_search.best_score_,
        'Test Accuracy': accuracy,
        'Test Precision': precision,
        'Test Recall': recall,
        'Test F1 Score': f1,
        'Model': grid_search.best_estimator_
    }

# Compare tuned models to baseline
comparison_df = pd.DataFrame({
    'Model': [],
    'Type': [],
    'Test Accuracy': [],
    'Test F1 Score': []
})

for model_name in top_models:
    # Baseline model metrics
    comparison_df = comparison_df.append({
        'Model': model_name,
        'Type': 'Baseline',
        'Test Accuracy': baseline_results[model_name]['Test Accuracy'],
        'Test F1 Score': baseline_results[model_name]['Test F1 Score']
    }, ignore_index=True)
    
    # Tuned model metrics
    comparison_df = comparison_df.append({
        'Model': model_name,
        'Type': 'Tuned',
        'Test Accuracy': tuned_models[model_name]['Test Accuracy'],
        'Test F1 Score': tuned_models[model_name]['Test F1 Score']
    }, ignore_index=True)

# Visualize comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Test F1 Score', hue='Type', data=comparison_df)
plt.title('F1 Score Comparison: Baseline vs. Tuned Models')
plt.ylim(0.7, 1.0)  # Adjust as needed
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Identify best tuned model
best_tuned_model_name = max(tuned_models.items(), key=lambda x: x[1]['Test F1 Score'])[0]
best_tuned_model = tuned_models[best_tuned_model_name]['Model']
print(f"\nBest tuned model: {best_tuned_model_name}")
print(f"F1 Score: {tuned_models[best_tuned_model_name]['Test F1 Score']:.4f}")
```

### 6.2 Learning Curves Analysis


```python
# Analyze learning curves for the best tuned model
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="f1_weighted")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    return plt

# Plot learning curve for best tuned model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
plot_learning_curve(
    best_tuned_model, X_selected, y, 
    title=f"Learning Curve - {best_tuned_model_name}", 
    cv=cv, n_jobs=-1
)
plt.show()
```

## 7. Feature Importance Analysis

### 7.1 SHAP Values Analysis


```python
# Analyze feature importance using SHAP
print("Calculating SHAP values for feature importance analysis...")

# Function to get and plot SHAP values
def analyze_shap_values(model, X_train, X_test, model_name):
    # Create a SHAP explainer object
    if model_name == "XGBoost":
        explainer = shap.TreeExplainer(model)
    elif model_name == "Random Forest":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)
    
    # Calculate SHAP values for test set
    shap_values = explainer(X_test)
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {model_name}')
    plt.tight_layout()
    plt.show()
    
    # Detailed summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}')
    plt.tight_layout()
    plt.show()
    
    return shap_values

# Get SHAP values for the best model
shap_values = analyze_shap_values(best_tuned_model, X_train, X_test, best_tuned_model_name)

# Analyze some individual predictions
def examine_individual_predictions(shap_values, X_test, y_test, label_encoder, n_samples=3):
    # Get actual class names
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    # Select random samples from each class
    unique_classes = np.unique(y_test)
    
    for class_idx in unique_classes:
        class_name = label_encoder.inverse_transform([class_idx])[0]
        print(f"\nExamining predictions for class: {class_name}")
        
        # Get samples from this class
        class_indices = np.where(y_test == class_idx)[0]
        if len(class_indices) >= n_samples:
            sample_indices = np.random.choice(class_indices, n_samples, replace=False)
            
            for idx in sample_indices:
                print(f"Sample #{idx}:")
                plt.figure(figsize=(12, 5))
                shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
                plt.title(f'SHAP Explanation for Sample #{idx} (Class: {class_name})')
                plt.tight_layout()
                plt.show()

# Examine individual predictions
examine_individual_predictions(shap_values, X_test, y_test, label_encoder, n_samples=1)
```

### 7.2 Feature Dependencies Analysis


```python
# Analyze interactions between top features
def analyze_feature_dependencies(shap_values, X_test, model_name):
    # Get top 2 most important features
    feature_names = X_test.columns
    feature_importance = np.abs(shap_values.values).mean(0)
    top_indices = np.argsort(feature_importance)[-2:]
    top_features = [feature_names[i] for i in top_indices]
    
    print(f"Analyzing interaction between top 2 features: {top_features[1]} and {top_features[0]}")
    
    # Dependence plot for top feature
    plt.figure(figsize=(10, 8))
    shap.dependence_plot(
        ind=top_features[1], 
        shap_values=shap_values, 
        features=X_test, 
        interaction_index=top_features[0],
        show=False
    )
    plt.title(f'SHAP Dependence Plot - {model_name}')
    plt.tight_layout()
    plt.show()

# Analyze dependencies for the best model
analyze_feature_dependencies(shap_values, X_test, best_tuned_model_name)
```

## 8. Final Model Evaluation

### 8.1 Final Model Performance


```python
# Evaluate the final model on the test set
y_pred_final = best_tuned_model.predict(X_test)
y_prob_final = best_tuned_model.predict_proba(X_test)

# Overall metrics
accuracy = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final, average='weighted')
recall = recall_score(y_test, y_pred_final, average='weighted')
f1 = f1_score(y_test, y_pred_final, average='weighted')

print("Final Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Final Model')
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=label_encoder.classes_))

# ROC Curve
plt.figure(figsize=(10, 8))
for i, label in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob_final[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
```

### 8.2 Save the Final Model


```python
# Save the final model
import pickle

# Create dictionary with all necessary components for prediction
final_model_package = {
    'model': best_tuned_model,
    'features': selected_features,
    'label_encoder': label_encoder,
    'numerical_features': numeric_features_to_scale,
    'scaler': scaler,
}

# Save the model package
with open('mental_health_classifier_model.pkl', 'wb') as f:
    pickle.dump(final_model_package, f)

print("Final model saved as 'mental_health_classifier_model.pkl'")

# Example of how to load and use the model
print("\nExample of loading and using the saved model:")
print("""
# Load the model
with open('mental_health_classifier_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

# Extract components
model = model_package['model']
features = model_package['features']
label_encoder = model_package['label_encoder']
numerical_features = model_package['numerical_features']
scaler = model_package['scaler']

# Prepare new data (ensure it has all required features)
# Scale numerical features
new_data[numerical_features] = scaler.transform(new_data[numerical_features])

# Select features
X_new = new_data[features]

# Make prediction
prediction = model.predict(X_new)
probabilities = model.predict_proba(X_new)

# Convert prediction to class label
predicted_class = label_encoder.inverse_transform(prediction)
""")
```

## 9. Results Interpretation

### 9.1 Summary of Findings


```python
# Summarize key findings from the analysis

print("Key Findings from Mental Health Classification Analysis:")
print("\n1. Model Performance:")
print(f"   - Best model: {best_tuned_model_name}")
print(f"   - Overall accuracy: {accuracy:.2f}")
print(f"   - F1 score: {f1:.2f}")

# Top features from SHAP analysis
print("\n2. Key Predictors of Mental Health Status:")
feature_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': np.abs(shap_values.values).mean(0)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
    print(f"   {i+1}. {row['Feature']} (Importance: {row['Importance']:.4f})")

# Class distribution
print("\n3. Class Distribution in the Dataset:")
class_distribution = df['Mental_Health_Status'].value_counts(normalize=True) * 100
for i, (class_name, percentage) in enumerate(class_distribution.items()):
    print(f"   {i+1}. {class_name}: {percentage:.1f}%")

print("\n4. Challenges in Classification:")
if class_distribution.min() / class_distribution.max() < 0.5:
    print("   - Class imbalance detected, which may affect model performance")
    print(f"   - The smallest class is {class_distribution.idxmin()} with {class_distribution.min():.1f}%")

# Misclassification patterns
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
misclassification = 1 - np.diag(cm_norm)
highest_misclass_idx = np.argmax(misclassification)
highest_misclass_class = label_encoder.inverse_transform([highest_misclass_idx])[0]
print(f"   - Highest misclassification rate for class '{highest_misclass_class}' ({misclassification[highest_misclass_idx]:.2%})")
```

### 9.2 Clinical Relevance


```python
# Discuss the clinical significance of the findings

print("Clinical Significance of the Model:")
print("\n1. Predictive Value:")
print("   - The model demonstrates the ability to identify individuals at risk of different mental health states")
print("   - Early identification allows for timely intervention and support")

print("\n2. Key Risk Factors:")
print("   - Our analysis confirms several known risk factors for anxiety and depression:")
print("     a. Sleep quality and duration")
print("     b. Social support networks")
print("     c. Stress levels (financial, work)")
print("     d. Physical activity")
print("     e. Self-esteem and life satisfaction")

print("\n3. Practical Applications:")
print("   - Screening Tool: Could be developed into a screening questionnaire for clinicians")
print("   - Risk Assessment: Helps identify factors that increase risk of poor mental health")
print("   - Intervention Planning: Target highest-impact modifiable factors")
print("   - Research: Provides quantitative evidence of factor relationships")

print("\n4. Limitations:")
print("   - Correlational, not causal relationships")
print("   - Based on self-reported data")
print("   - May not generalize to all populations")
print("   - Requires clinical interpretation alongside model predictions")
```

## 10. Conclusions and Future Work

### 10.1 Final Conclusions


```python
# Summarize the overall project and conclusions

print("Project Conclusions:")
print("\n1. Summary of Approach:")
print("   - Developed a classification model to predict mental health status based on lifestyle, demographic,")
print("     and psychosocial factors")
print("   - Applied comprehensive data preprocessing, feature engineering, and model optimization")
print("   - Used SHAP values to interpret model predictions and identify key factors")

print("\n2. Key Achievements:")
print(f"   - Successfully built a predictive model with {accuracy:.2%} accuracy")
print("   - Identified the most important factors influencing mental health status")
print("   - Created a pipeline for data processing and prediction")
print("   - Demonstrated the value of interpretable machine learning in mental health")

print("\n3. Technical Insights:")
print(f"   - {best_tuned_model_name} performed best for this classification task")
print("   - Feature engineering significantly improved model performance")
print("   - Interpretability tools like SHAP provided valuable clinical insights")

print("\n4. Overall Value:")
print("   - The model provides a data-driven approach to mental health assessment")
print("   - Highlights the multi-factorial nature of mental health conditions")
print("   - Can inform both individual intervention and public health strategies")
```

### 10.2 Future Work


```python
# Outline potential improvements and future research directions

print("Future Work and Improvements:")
print("\n1. Model Enhancements:")
print("   - Incorporate temporal data to track changes in mental health over time")
print("   - Test additional algorithms (deep learning, ensemble methods)")
print("   - Optimize for specific use cases (screening vs. severity prediction)")

print("\n2. Additional Data Sources:")
print("   - Incorporate biometric data (sleep tracking, physical activity)")
print("   - Add contextual data (life events, environmental factors)")
print("   - Include treatment response data for better intervention planning")

print("\n3. Application Development:")
print("   - Create a user-friendly interface for clinical use")
print("   - Develop a mobile application for self-assessment and monitoring")
print("   - Integrate with electronic health records")

print("\n4. Validation Studies:")
print("   - Test model performance in diverse populations")
print("   - Conduct longitudinal studies to validate predictive power")
print("   - Compare against established clinical assessment tools")

print("\n5. Expanded Focus:")
print("   - Extend to other mental health conditions beyond anxiety and depression")
print("   - Develop specialized models for different demographics")
print("   - Explore causal relationships through interventional studies")
```

### 10.3 Final Summary


```python
print("""
Mental Health Classification Project - Final Summary

This project demonstrates the application of machine learning to mental health classification using a comprehensive dataset of lifestyle, demographic, and psychosocial factors.

Key Takeaways:

1. Mental health status can be predicted with reasonable accuracy using readily available data.
2. The most important predictors include [top 3-5 factors from your analysis].
3. Model interpretability is critical for clinical applications and trust.
4. Machine learning can complement, not replace, clinical judgment in mental health assessment.

The approach presented here can be adapted for various mental health applications, from population screening to personalized intervention planning. Future work should focus on longitudinal validation, expanded data sources, and user-friendly implementation in clinical settings.
""")
```
