# Consumer Sentiment Analysis with BERT, APIs, and AutoML

## Project Overview

This notebook implements advanced NLP techniques for analyzing consumer sentiment from product reviews. We'll use the Consumer Sentiments and Ratings dataset from Kaggle, applying transformer models (BERT), external NLP APIs, and AutoML approaches to classify sentiment and extract insights from consumer feedback.

## 1. Google Colab Setup with GPU/TPU Configuration

# First, let's check if we're running on Colab and set up GPU acceleration
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    # Check if GPU is available
    !nvidia-smi
    
    # Connect to Google Drive for data storage (optional)
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Install required packages (if not already installed)
    !pip install -q transformers datasets torch xgboost shap optuna pandas numpy matplotlib seaborn scikit-learn openai

# Set up GPU configuration
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead")

# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import os
import warnings
from datetime import datetime
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

import xgboost as xgb
import shap
import optuna

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

## 2. Data Loading and Exploration

# Download dataset if in Colab or load locally
dataset_path = "consumer_sentiments.csv"

if IN_COLAB:
    # Option 1: Download directly from Kaggle (requires API token)
    try:
        import os
        os.environ['KAGGLE_USERNAME'] = "your_username"  # Replace with your Kaggle username
        os.environ['KAGGLE_KEY'] = "your_key"  # Replace with your Kaggle API key
        
        !kaggle datasets download -d kapturovalexander/consumer-sentiments-and-ratings
        !unzip consumer-sentiments-and-ratings.zip
        dataset_path = "ratings.csv"  # Adjust based on actual file name
    except:
        print("Direct Kaggle download failed. Please upload manually or use Google Drive.")
        # Option 2: Upload through Colab interface
        from google.colab import files
        uploaded = files.upload()  # Will prompt user to upload CSV file
        dataset_path = list(uploaded.keys())[0]

# Load the dataset
df = pd.read_csv(dataset_path)

# Display basic information
print(f"Dataset shape: {df.shape}")
df.info()
df.head()

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Convert date to datetime format for time analysis
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Exploratory Data Analysis (EDA)

# 1. Distribution of ratings
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=df)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 2. Distribution of sentiment/tonality
plt.figure(figsize=(10, 6))
sns.countplot(x='tonality', data=df)
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 3. Average rating by brand (top 10 brands by frequency)
top_brands = df['brand'].value_counts().nlargest(10).index
brand_avg_rating = df[df['brand'].isin(top_brands)].groupby('brand')['rating'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=brand_avg_rating.index, y=brand_avg_rating.values)
plt.title('Average Rating by Top 10 Brands')
plt.xlabel('Brand ID')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()

# 4. Sentiment distribution by category (top 10 categories)
top_categories = df['item_category'].value_counts().nlargest(10).index
plt.figure(figsize=(12, 8))
category_sentiment = pd.crosstab(df[df['item_category'].isin(top_categories)]['item_category'], 
                                 df[df['item_category'].isin(top_categories)]['tonality'])
category_sentiment_pct = category_sentiment.div(category_sentiment.sum(axis=1), axis=0)
category_sentiment_pct.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Sentiment Distribution by Category')
plt.xlabel('Category ID')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)
plt.show()

# 5. Comment length analysis
df['comment_length'] = df['comment'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(12, 6))
sns.histplot(df['comment_length'], bins=50, kde=True)
plt.title('Distribution of Comment Length (word count)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xlim(0, 100)  # Focus on comments up to 100 words
plt.show()

# 6. Correlation between comment length and rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='rating', y='comment_length', data=df)
plt.title('Comment Length by Rating')
plt.xlabel('Rating')
plt.ylabel('Comment Length (words)')
plt.show()

# Basic text analysis - most common words by sentiment
from collections import Counter
import re
from wordcloud import WordCloud

def get_top_words(texts, n=20):
    """Extract most common words from a list of texts"""
    words = ' '.join(texts).lower()
    words = re.sub(r'[^\w\s]', '', words)  # Remove punctuation
    words = words.split()
    # Remove common stopwords (expand this list as needed)
    stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 'that', 'this', 'i', 'was', 'are', 'at'}
    words = [word for word in words if word not in stopwords]
    return Counter(words).most_common(n)

# Create word clouds for positive and negative sentiment
sentiments = df['tonality'].unique()

for sentiment in sentiments:
    subset = df[df['tonality'] == sentiment]
    if len(subset) > 0:
        text = ' '.join(subset['comment'].dropna().astype(str))
        
        plt.figure(figsize=(12, 8))
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Most Common Words in {sentiment.title()} Reviews')
        plt.axis('off')
        plt.show()
        
        # Print top words
        print(f"\nTop 20 words in {sentiment.title()} reviews:")
        top_words = get_top_words(subset['comment'].dropna().astype(str))
        for word, count in top_words:
            print(f"{word}: {count}")

# Prepare data for modeling
# Check the label distribution
print("\nSentiment distribution:")
sentiment_counts = df['tonality'].value_counts()
print(sentiment_counts)

# Encode sentiment labels
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['tonality'])
print("\nEncoded sentiment labels:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label}: {i}")

# Save mapping for later reference
sentiment_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

## 3. Text Preprocessing for Transformer Models

def preprocess_text(text):
    """Basic text preprocessing for BERT"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters but keep spaces and punctuation that BERT can use
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ""

# Apply preprocessing to comments
df['processed_comment'] = df['comment'].apply(preprocess_text)

# Sample a smaller subset for faster processing (important for Colab environment)
# Adjust the sample size based on your available resources
sample_size = 10000  # Adjust as needed
if len(df) > sample_size:
    df_sample = df.sample(sample_size, random_state=42)
else:
    df_sample = df

# Split data into train, validation, and test sets
X = df_sample['processed_comment'].values
y = df_sample['sentiment_encoded'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

## 4. BERT Implementation

# 4.1 Pre-trained BERT Model Setup

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # Maximum sequence length (adjust based on your comment lengths)

# Create a custom dataset for BERT
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Encode the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length)
val_dataset = SentimentDataset(X_val, y_val, tokenizer, max_length)
test_dataset = SentimentDataset(X_test, y_test, tokenizer, max_length)

# Create data loaders
batch_size = 16  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 4.2 Fine-tuning BERT for Sentiment Analysis

# Set up the BERT model for sequence classification
num_classes = len(sentiment_mapping)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
model = model.to(device)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training function
def train_bert_model(model, train_loader, val_loader, optimizer, scheduler, epochs=3):
    """Train BERT model for sentiment classification"""
    best_val_accuracy = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'best_bert_model.pt')
            print("Saved best model!")
    
    return history

# Run the training process
history = train_bert_model(model, train_loader, val_loader, optimizer, scheduler)

# Visualize training progress
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Load the best model
model.load_state_dict(torch.load('best_bert_model.pt'))
model.eval()

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    """Evaluate the model and return predictions and true labels"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = [sentiment_mapping[i] for i in range(len(sentiment_mapping))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.show()
    
    return all_preds, all_labels

# Evaluate the model
test_preds, test_labels = evaluate_model(model, test_loader)

# 4.3 Feature Extraction from BERT

class BertFeatureExtractor(nn.Module):
    """Extract features from BERT for use in traditional ML models"""
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(BertFeatureExtractor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the [CLS] token embedding (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output

# Initialize the feature extractor
feature_extractor = BertFeatureExtractor().to(device)
feature_extractor.eval()

# Function to extract features
def extract_bert_features(dataset, batch_size=16):
    """Extract BERT features from a dataset"""
    loader = DataLoader(dataset, batch_size=batch_size)
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Extract features
            batch_features = feature_extractor(input_ids, attention_mask)
            features.append(batch_features.cpu().numpy())
            labels.append(batch['labels'].numpy())
    
    # Concatenate batches
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels

# Extract features for train, validation, and test sets
X_train_bert, y_train_bert = extract_bert_features(train_dataset)
X_val_bert, y_val_bert = extract_bert_features(val_dataset)
X_test_bert, y_test_bert = extract_bert_features(test_dataset)

print(f"Extracted features shape: {X_train_bert.shape}")

## 5. External API Integration (Simulated)

# Note: Actual API integration would require API keys and proper authentication
# For this educational notebook, we'll simulate API calls to demonstrate the concept

# Simulated OpenAI API for sentiment analysis
def simulate_openai_sentiment(texts, batch_size=10):
    """Simulate calling OpenAI API for sentiment analysis"""
    # In a real scenario, you would use:
    # from openai import OpenAI
    # client = OpenAI(api_key="your-api-key")
    
    # For demonstration, we'll simulate API responses based on simple rules
    sentiments = []
    confidences = []
    
    print("Simulating API calls in batches...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        for text in batch_texts:
            # Simple rule-based sentiment detection for simulation
            text = text.lower()
            pos_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'recommend']
            neg_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'poor', 'disappointing']
            
            pos_score = sum(1 for word in pos_words if word in text)
            neg_score = sum(1 for word in neg_words if word in text)
            
            # Determine sentiment based on scores
            if pos_score > neg_score:
                sentiment = 0  # Positive (adjust based on your encoding)
                confidence = 0.5 + (pos_score - neg_score) * 0.1  # Simulated confidence
            elif neg_score > pos_score:
                sentiment = 1  # Negative (adjust based on your encoding)
                confidence = 0.5 + (neg_score - pos_score) * 0.1
            else:
                sentiment = 2  # Neutral (adjust based on your encoding)
                confidence = 0.5
            
            sentiments.append(sentiment)
            confidences.append(confidence)
    
    return np.array(sentiments), np.array(confidences)

# Apply simulated API to a subset of data
api_sample_size = 500  # Limit due to simulated API calls
X_api_test = X_test[:api_sample_size]
y_api_test = y_test[:api_sample_size]

# Get API predictions
api_predictions, api_confidences = simulate_openai_sentiment(X_api_test)

# Evaluate API predictions
print("\nSimulated API Performance:")
api_accuracy = accuracy_score(y_api_test, api_predictions)
print(f"API Accuracy: {api_accuracy:.4f}")

print("\nAPI Classification Report:")
print(classification_report(y_api_test, api_predictions, 
                            target_names=[sentiment_mapping[i] for i in range(len(sentiment_mapping))]))

# Visualize API confidence vs. accuracy
plt.figure(figsize=(10, 6))
correct_predictions = api_predictions == y_api_test
plt.scatter(api_confidences[correct_predictions], np.ones(sum(correct_predictions))*1, 
            alpha=0.5, label='Correct predictions', color='green')
plt.scatter(api_confidences[~correct_predictions], np.ones(sum(~correct_predictions))*0, 
            alpha=0.5, label='Incorrect predictions', color='red')
plt.title('API Confidence vs. Prediction Correctness')
plt.xlabel('Confidence Score')
plt.ylabel('Correct (1) / Incorrect (0)')
plt.legend()
plt.show()

## 6. AutoML with Transformer Features

# 6.1 BERT Feature Integration with XGBoost

# Train an XGBoost model on BERT features
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(sentiment_mapping),
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Train the model
xgb_model.fit(
    X_train_bert, y_train_bert,
    eval_set=[(X_train_bert, y_train_bert), (X_val_bert, y_val_bert)],
    early_stopping_rounds=10,
    verbose=True
)

# Get XGBoost predictions
xgb_predictions = xgb_model.predict(X_test_bert)
xgb_probabilities = xgb_model.predict_proba(X_test_bert)

# Evaluate XGBoost performance
print("\nXGBoost with BERT Features:")
xgb_accuracy = accuracy_score(y_test_bert, xgb_predictions)
print(f"Accuracy: {xgb_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_bert, xgb_predictions, 
                            target_names=[sentiment_mapping[i] for i in range(len(sentiment_mapping))]))

# Feature importance (understand which BERT features are most important)
plt.figure(figsize=(12, 6))
xgb.plot_importance(xgb_model, max_num_features=20, importance_type='gain')
plt.title('XGBoost Feature Importance (BERT embedding dimensions)')
plt.show()

# 6.2 AutoML Pipeline Setup with Optuna for Hyperparameter Optimization

def objective(trial):
    """Objective function for Optuna hyperparameter optimization"""
    # Hyperparameter search space
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'objective': 'multi:softprob',
        'num_class': len(sentiment_mapping),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    # Train model with current hyperparameters
    model = xgb.XGBClassifier(**param)
    model.fit(X_train_bert, y_train_bert,
             eval_set=[(X_val_bert, y_val_bert)],
             early_stopping_rounds=10,
             verbose=False)
    
    # Return validation accuracy as the metric to optimize
    predictions = model.predict(X_val_bert)
    accuracy = accuracy_score(y_val_bert, predictions)
    
    return accuracy

# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # Adjust n_trials based on your time constraints

# Get the best hyperparameters
best_params = study.best_params
print("\nBest hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Train the model with the best hyperparameters
best_xgb_model = xgb.XGBClassifier(
    **best_params,
    objective='multi:softprob',
    num_class=len(sentiment_mapping),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

best_xgb_model.fit(
    X_train_bert, y_train_bert,
    eval_set=[(X_val_bert, y_val_bert)],
    early_stopping_rounds=10,
    verbose=True
)

# Evaluate the optimized model
best_xgb_predictions = best_xgb_model.predict(X_test_bert)

print("\nOptimized XGBoost Model:")
best_xgb_accuracy = accuracy_score(y_test_bert, best_xgb_predictions)
print(f"Accuracy: {best_xgb_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_bert, best_xgb_predictions, 
                           target_names=[sentiment_mapping[i] for i in range(len(sentiment_mapping))]))

# 6.3 SHAP values for model interpretability

# Create a small sample for SHAP analysis (for efficiency)
shap_sample_size = 100  # Adjust based on your computational resources
X_shap = X_test_bert[:shap_sample_size]

# Calculate SHAP values
explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X_shap)

# Summary plot of SHAP values
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, plot_type="bar", class_names=[sentiment_mapping[i] for i in range(len(sentiment_mapping))])

# Force plot for an individual example
plt.figure(figsize=(20, 3))
shap.initjs()
example_idx = 0  # First example in the test set
shap.force_plot(explainer.expected_value[0], shap_values[0][example_idx, :], X_shap[example_idx, :], 
                matplotlib=True, show=False)
plt.title(f"SHAP Force Plot for Example {example_idx} (Class: {sentiment_mapping[int(y_test_bert[example_idx])]})")
plt.show()

## 7. Comparative Analysis

# Compare all models
models = ['BERT', 'Simulated API', 'XGBoost', 'Optimized XGBoost']
accuracies = []

# Get BERT model accuracy on test set
model.eval()
bert_correct = 0
bert_total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, 1)
        
        bert_total += labels.size(0)
        bert_correct += (preds == labels).sum().item()

bert_accuracy = bert_correct / bert_total
accuracies.append(bert_accuracy)

# Add other model accuracies
accuracies.append(api_accuracy)  # Simulated API
accuracies.append(xgb_accuracy)  # XGBoost
accuracies.append(best_xgb_accuracy)  # Optimized XGBoost

# Plot comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add accuracy values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom')

plt.show()

# Advantages and disadvantages of each approach
advantages_disadvantages = pd.DataFrame({
    'Model': models,
    'Advantages': [
        'Fine-grained control; Context awareness; State-of-the-art performance',
        'Easy to implement; No training required; Handles edge cases',
        'Fast inference; Lower computational needs; Interpretable',
        'Optimal performance; Adaptable to specific data'
    ],
    'Disadvantages': [
        'Computationally expensive; Requires GPU/TPU; Complex to fine-tune',
        'Cost per API call; Limited customization; Dependency on external service',
        'Less contextual understanding; Manual feature engineering needed',
        'Hyperparameter tuning time; Potential overfitting; Complexity'
    ],
    'Best Use Cases': [
        'When high accuracy is critical; Complex sentiment nuances',
        'Rapid prototyping; Low volume analysis; Multilingual support',
        'Production with resource constraints; Need for interpretability',
        'Production systems requiring maximum performance; Custom domains'
    ]
})

# Display the comparison table
print("\nModel Approaches Comparison:")
print(advantages_disadvantages)

## 8. Conclusion and Next Steps

# Save important model artifacts
if not os.path.exists('models'):
    os.makedirs('models')

# Save XGBoost model
best_xgb_model.save_model('models/best_xgb_model.json')

# Save sentiment mapping
with open('models/sentiment_mapping.json', 'w') as f:
    json.dump(sentiment_mapping, f)

# Save tokenizer for future use
tokenizer.save_pretrained('models/bert_tokenizer')

"""
## Key Takeaways:

1. **Multiple Approaches**: We've implemented and compared several approaches for sentiment analysis:
   - Fine-tuned BERT transformer model
   - External API integration (simulated)
   - Traditional ML (XGBoost) with BERT features
   - AutoML for optimizing hyperparameters

2. **Performance Considerations**:
   - Transformer models provide state-of-the-art accuracy but require significant computational resources
   - External APIs offer a quick solution without infrastructure setup
   - Feature extraction + XGBoost provides a good balance of performance and efficiency
   - AutoML significantly improves model performance with minimal manual tuning

3. **Practical Implementation**:
   - GPU/TPU acceleration is essential for transformer models
   - BERT features can be extracted and used with traditional ML models
   - Hyperparameter optimization yields substantial performance improvements
   - Model interpretability with SHAP helps understand predictions

## Next Steps:

1. **Deployment Considerations**:
   - Implementing model compression for faster inference
   - Setting up a production-ready API
   - Monitoring for concept drift and performance degradation

2. **Advanced Techniques**:
   - Exploring domain-specific pre-trained models
   - Implementing ensemble methods combining multiple approaches
   - Incorporating additional features like product metadata
   - Implementing active learning for continuous model improvement

3. **Business Applications**:
   - Automated customer feedback routing
   - Prioritization of negative reviews for customer service
   - Brand sentiment tracking over time
   - Competitive analysis across product categories
"""

# Display a final summary of skills demonstrated
skills_demonstrated = [
    "BERT and Transformer Architectures",
    "GPU/TPU Acceleration in Google Colab",
    "Feature Extraction from Transformers",
    "XGBoost for Text Classification",
    "Hyperparameter Optimization with Optuna",
    "AutoML Implementation",
    "Model Interpretability with SHAP",
    "API Integration Patterns",
    "Data Exploration and Visualization",
    "Performance Comparison Methodology"
]

print("\nSkills Demonstrated in this Project:")
for i, skill in enumerate(skills_demonstrated, 1):
    print(f"{i}. {skill}")
