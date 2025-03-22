# Consumer Sentiment Analysis with BERT, APIs, and AutoML

## Overview
This project focuses on leveraging BERT, external NLP APIs, and AutoML techniques to analyze consumer sentiment from product reviews. It utilizes the [Consumer Sentiments and Ratings](https://www.kaggle.com/datasets/kapturovalexander/consumer-sentiments-and-ratings) dataset from Kaggle and is designed to run in Google Colab with GPU/TPU acceleration.

## Project Goals
- Apply transformer models and external APIs to consumer sentiment analysis
- Integrate AutoML approaches with BERT-based feature extraction
- Compare performance of various approaches: custom BERT, external APIs, and AutoML
- Optimize models for processing large volumes of consumer review data

## Skills Covered
- Google Colab environment with GPU/TPU acceleration
- Hugging Face Transformers library
- BERT and other transformer architectures for sentiment analysis
- AutoML frameworks with custom transformers
- XGBoost with NLP features
- Hyperparameter optimization for transformer models

## Dataset Description
The dataset contains customer reviews for various products with details about categories, brands, ratings, and sentiment analysis. Key columns include:
- item_category: Category ID of the reviewed product
- item_id: Unique identifier for the product
- brand: Brand ID associated with the product
- user_id: Identifier for the reviewer
- date: Review posting date
- comment: Text content of the review
- rating: Numerical rating (typically 1-5)
- tonality: Sentiment classification (positive, negative, etc.)

## Project Structure
1. Google Colab Setup with GPU/TPU Configuration
2. Data Loading and Exploration
3. Text Preprocessing for Transformer Models
4. BERT Implementation:
   - Pre-trained BERT Model Setup
   - Fine-tuning for Consumer Sentiment
   - Feature Extraction from BERT
5. External API Integration:
   - For simplicity: OpenAI API
6. AutoML with Transformer Features:
   - BERT Feature Integration with XGBoost
   - AutoML Pipeline Setup
   - Hyperparameter Optimization
7. Comparative Analysis:
   - Custom BERT vs. External APIs vs. AutoML
   - Performance Metrics Across Methods

## Expected Outcomes
- Hands-on experience with transformer models in Google Colab
- Practical understanding of NLP API integration
- Ability to combine transformer models with AutoML approaches
- Skills in optimizing NLP models for large datasets
- Comparative knowledge of when to use each approach
