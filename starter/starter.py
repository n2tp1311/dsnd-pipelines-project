#%% md
# # Pipeline Project
#%% md
# You will be using the provided data to create a machine learning model pipeline.
# 
# You must handle the data appropriately in your pipeline to predict whether an
# item is recommended by a customer based on their review.
# Note the data includes numerical, categorical, and text data.
# 
# You should ensure you properly train and evaluate your model.
#%% md
# ## The Data
#%% md
# The dataset has been anonymized and cleaned of missing values.
# 
# There are 8 features for to use to predict whether a customer recommends or does
# not recommend a product.
# The `Recommended IND` column gives whether a customer recommends the product
# where `1` is recommended and a `0` is not recommended.
# This is your model's target/
#%% md
# The features can be summarized as the following:
# 
# - **Clothing ID**: Integer Categorical variable that refers to the specific piece being reviewed.
# - **Age**: Positive Integer variable of the reviewers age.
# - **Title**: String variable for the title of the review.
# - **Review Text**: String variable for the review body.
# - **Positive Feedback Count**: Positive Integer documenting the number of other customers who found this review positive.
# - **Division Name**: Categorical name of the product high level division.
# - **Department Name**: Categorical name of the product department name.
# - **Class Name**: Categorical name of the product class name.
# 
# The target:
# - **Recommended IND**: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
#%% md
# ## Import 
#%%
# system
import os 
import multiprocessing 
from joblib import (
    parallel_backend,
    Parallel,
    delayed
)
# warnings
import warnings

# Data manipulation and analysis
import pandas as pd
import numpy as np
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Text processing and ML
import spacy
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Settings
plt.style.use('ggplot')
warnings.filterwarnings('ignore')
%matplotlib inline

# Random seed for reproducibility
RANDOM_STATE = 27

# Set environment variables for optimization
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['VECLIB_MAXIMUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Building the Machine Learning Pipeline with Performance Optimization
print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
#%% md
# ## Load Data
#%%
# Load data
df = pd.read_csv(
    'data/reviews.csv',
)

df.info()
df.head()
#%% md
# ## Preparing features (`X`) & target (`y`)
#%%
data = df

# separate features from labels
X = data.drop('Recommended IND', axis=1)
y = data['Recommended IND'].copy()

print('Labels:', y.unique())
print('Features:')
display(X.head())
#%%
# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    shuffle=True,
    random_state=RANDOM_STATE,
)
#%% md
# # Your Work
#%% md
# ## Set cols Variable
#%%
cols_feature_num = ['Age', 'Positive Feedback Count']
cols_feature_cat = ['Division Name', 'Department Name', 'Class Name'] 
cols_feature_text = ['Title', 'Review Text']
cols_feature = cols_feature_num + cols_feature_cat + cols_feature_text

col_target = 'Recommended IND'
#%% md
# 
# ## Data Exploration
#%% md
# #### Target analysis
#%%
# Plot target distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x=col_target)
plt.title('Distribution of Recommendations')
plt.show()

# Print distribution percentages
rec_dist = df[col_target].value_counts(normalize=True)
print("\nRecommendation Distribution:")
print(rec_dist)
#%% md
# ### Numerical Features Analysis
#%% md
# ##### Distribution
#%%
# Create plots for each numerical feature
for col in cols_feature_num:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Distribution plot
    sns.histplot(data=df, x=col, ax=ax1)
    ax1.set_title(f'Distribution of {col}')
    
    # Box plot by recommendation
    sns.boxplot(data=df, x=col_target, y=col, ax=ax2)
    ax2.set_title(f'{col} by Recommendation')
    
    # Violin plot to show distribution density
    sns.violinplot(data=df, x=col_target, y=col, ax=ax3)
    ax3.set_title(f'Distribution of {col} by Recommendation')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics by recommendation
    print(f"\n{col} Summary Statistics by Recommendation:")
    print(df.groupby(col_target)[col].describe())
#%% md
# #### Correlation 
#%%
# Create correlation matrix for numerical features
numerical_data = df[cols_feature_num + [col_target]]
correlation_matrix = numerical_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
#%% md
# ### Categorical Features Analysis
#%% md
# #### Distribution
#%%
for col in cols_feature_cat:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Get value counts and recommendation rates
    value_counts = df[col].value_counts()
    rec_rate = df.groupby(col)['Recommended IND'].mean()
    
    # Sort both by value counts (most frequent first)
    sorted_index = value_counts.index
    value_counts = value_counts[sorted_index]
    rec_rate = rec_rate[sorted_index]
    
    # Distribution of categories
    sns.barplot(x=value_counts.values, y=sorted_index, ax=ax1)
    ax1.set_title(f'Distribution of {col}')
    ax1.set_xlabel('Count')
    
    # Recommendation rate by category
    sns.barplot(x=rec_rate.values, y=sorted_index, ax=ax2)
    ax2.set_title(f'Recommendation Rate by {col}')
    ax2.set_xlabel('Recommendation Rate')
    
    plt.tight_layout()
    plt.show()
    
    # Print value counts and recommendation rates
    print(f"\n{col} Value Counts:")
    print(value_counts)
    print(f"\n{col} Recommendation Rates:")
    print(rec_rate)
#%% md
# #### Chi-square
#%%
# Calculate chi-square test of independence for categorical features
from scipy.stats import chi2_contingency

print("\nChi-square test results for categorical features:")
for col in cols_feature_cat:
    contingency_table = pd.crosstab(df[col], df[col_target])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"\n{col}:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.10f}")
#%% md
# ### Text analysis
#%% md
# #### Text length 
#%%
 print("Text Length Statistics:")
df['Title_length'] = df['Title'].str.len()
df['Review_length'] = df['Review Text'].str.len()

for col in ['Title_length', 'Review_length']:
    plt.figure(figsize=(15, 5))
    
    # Distribution plot
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    
    # Box plot by recommendation
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='Recommended IND', y=col)
    plt.title(f'{col} by Recommendation')
    
    # Average length by recommendation
    plt.subplot(1, 3, 3)
    sns.barplot(data=df, x='Recommended IND', y=col)
    plt.title(f'Average {col} by Recommendation')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{col} Summary Statistics:")
    print(df[col].describe())
    print("\nMean length by recommendation:")
    print(df.groupby('Recommended IND')[col].mean())

#%% md
# #### Text preprocessing
#%%
# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_with_spacy(text):
    """Preprocess text using spaCy's pipeline."""
    if pd.isna(text):
        return ''
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Get lemmatized tokens, excluding stop words and punctuation
    tokens = [token.lemma_.lower() for token in doc 
             if not token.is_stop 
             and not token.is_punct 
             and token.lemma_.strip()]
    
    return ' '.join(tokens)

# Apply preprocessing
print("Preprocessing text data with spaCy...")
df['Title_processed'] = df['Title'].apply(preprocess_with_spacy)
df['Review_processed'] = df['Review Text'].apply(preprocess_with_spacy)

#%% md
#  #### Named Entity Recognition Analysis
#%%
def analyze_entities(texts, n=10):
    """Analyze named entities in a series of texts."""
    entity_counts = Counter()
    entity_types = Counter()
    
    for text in texts:
        if pd.isna(text):
            continue
        doc = nlp(text)
        for ent in doc.ents:
            entity_counts[ent.text.lower()] += 1
            entity_types[ent.label_] += 1
    
    return (pd.DataFrame(entity_counts.most_common(n), columns=['Entity', 'Count']),
            pd.DataFrame(entity_types.most_common(), columns=['Entity Type', 'Count']))

# Analyze entities by recommendation
for rec in [0, 1]:
    print(f"\nNamed Entity Analysis for {'not ' if rec == 0 else ''}recommended reviews:")
    
    # Analyze reviews
    entities, entity_types = analyze_entities(df[df['Recommended IND'] == rec]['Review Text'])
    
    # Plot entities
    plt.figure(figsize=(15, 5))
    
    # Most common entities
    plt.subplot(1, 2, 1)
    sns.barplot(data=entities, x='Count', y='Entity')
    plt.title(f"Most Common Entities in {'Not ' if rec == 0 else ''}Recommended Reviews")
    
    # Entity types
    plt.subplot(1, 2, 2)
    sns.barplot(data=entity_types, x='Count', y='Entity Type')
    plt.title(f"Entity Types in {'Not ' if rec == 0 else ''}Recommended Reviews")
    
    plt.tight_layout()
    plt.show()
    
    print("\nMost common entities:")
    print(entities)
    print("\nEntity types distribution:")
    print(entity_types)
#%% md
# #### Part-of-Speech Analysis
#%%
def analyze_pos(texts, n=10):
    """Analyze parts of speech in a series of texts."""
    pos_counts = Counter()
    important_words = Counter()
    
    for text in texts:
        if pd.isna(text):
            continue
        doc = nlp(text)
        for token in doc:
            pos_counts[token.pos_] += 1
            # Only count content words (nouns, verbs, adjectives, adverbs)
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop:
                important_words[token.text.lower()] += 1
    
    return (pd.DataFrame(pos_counts.most_common(), columns=['POS', 'Count']),
            pd.DataFrame(important_words.most_common(n), columns=['Word', 'Count']))

# Analyze POS by recommendation
for rec in [0, 1]:
    print(f"\nPart-of-Speech Analysis for {'not ' if rec == 0 else ''}recommended reviews:")
    
    pos_counts, important_words = analyze_pos(df[df['Recommended IND'] == rec]['Review Text'])
    
    plt.figure(figsize=(15, 5))
    
    # POS distribution
    plt.subplot(1, 2, 1)
    sns.barplot(data=pos_counts, x='Count', y='POS')
    plt.title(f"POS Distribution in {'Not ' if rec == 0 else ''}Recommended Reviews")
    
    # Important words
    plt.subplot(1, 2, 2)
    sns.barplot(data=important_words, x='Count', y='Word')
    plt.title(f"Most Common Content Words in {'Not ' if rec == 0 else ''}Recommended Reviews")
    
    plt.tight_layout()
    plt.show()
    
    print("\nPOS distribution:")
    print(pos_counts)
    print("\nMost common content words:")
    print(important_words)
#%% md
# ## Building Pipeline
#%% md
# ### Custom Transfomer
#%%
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re
from collections import Counter

class SimpleTextTransformer(BaseEstimator, TransformerMixin):
    """
    A simple text transformer that extracts basic text statistics and features.
    No external dependencies required.
    """
    def __init__(self):
        self.feature_names = [
            'text_length',
            'word_count',
            'avg_word_length',
            'sentence_count',
            'unique_words',
            'punctuation_count',
            'uppercase_words',
            'numbers_count',
            'special_chars_count'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for idx, text in X.iterrows():
            if not isinstance(text, str):
                text = str(text)
            
            # Basic text statistics
            text_length = len(text)
            words = text.split()
            word_count = len(words)
            avg_word_length = np.mean([len(word) for word in words]) if word_count > 0 else 0
            
            # Sentence count (simple split by common sentence endings)
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Unique words
            unique_words = len(set(words))
            
            # Punctuation count
            punctuation_count = len(re.findall(r'[.,!?;:]', text))
            
            # Uppercase words
            uppercase_words = len([word for word in words if word.isupper()])
            
            # Numbers count
            numbers_count = len(re.findall(r'\d+', text))
            
            # Special characters count
            special_chars_count = len(re.findall(r'[^a-zA-Z0-9\s]', text))
            
            features.append([
                text_length,
                word_count,
                avg_word_length,
                sentence_count,
                unique_words,
                punctuation_count,
                uppercase_words,
                numbers_count,
                special_chars_count
            ])
        
        return np.array(features)

    def get_feature_names_out(self):
        return self.feature_names
#%% md
# 
# 
# 
# 
# ### Preprocessing
#%%
import spacy

nlp = spacy.load('en_core_web_sm')
#%%
print("\nCreating preprocessing pipelines...")

# Numerical pipeline with imputer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline with imputer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])


print("\nCombining features in ColumnTransformer...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, cols_feature_num),
        ('cat', categorical_transformer, cols_feature_cat),
        ('text_title', SimpleTextTransformer(), ['Title']),
        ('text_review', SimpleTextTransformer(), ['Review Text'])
    ],
    remainder = 'drop',
)
#%% md
# ### Full pipeline
#%%
print("\nCreating the full pipeline...")
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
    ],
)

#%% md
# 
# ## Training Pipeline
#%%
base_score = cross_val_score(
    full_pipeline,
    X_train, 
    y_train, 
    scoring='accuracy',
    cv=5
).mean()
print(f"\nBase accuracy score: {base_score:.4f}")
#%% md
# ## Fine-Tuning Pipeline
#%%
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
import numpy as np

def create_objective(X_train, y_train, pipeline):
    def objective(trial):
        # Define the parameter space
        params = {
            'classifier__n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'classifier__max_depth': trial.suggest_int('max_depth', 5, 50),
            'classifier__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'classifier__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'classifier__max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'classifier__class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        }
        
        # Update pipeline with suggested parameters
        pipeline.set_params(**params)
        
        # Perform cross-validation
        scores = cross_val_score(
            pipeline, 
            X_train, 
            y_train, 
            cv=3, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        return scores.mean()
    
    return objective

def optimize_with_optuna(X_train, y_train, pipeline, n_trials=30):
    # Create study with TPE sampler
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Create objective function
    objective = create_objective(X_train, y_train, pipeline)
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: print(f"Trial {trial.number}: Accuracy = {trial.value:.4f}")
        ]
    )
    
    return study

# Run optimization
print("Starting Optuna optimization...")
study = optimize_with_optuna(X_train, y_train, full_pipeline, n_trials=30)

# Print results
print("\nBest trial:")
print(f"Value (Accuracy): {study.best_trial.value:.4f}")
print("\nBest parameters:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")

# Create best model
best_params = {
    'classifier__n_estimators': study.best_trial.params['n_estimators'],
    'classifier__max_depth': study.best_trial.params['max_depth'],
    'classifier__min_samples_split': study.best_trial.params['min_samples_split'],
    'classifier__min_samples_leaf': study.best_trial.params['min_samples_leaf'],
    'classifier__max_features': study.best_trial.params['max_features'],
    'classifier__class_weight': study.best_trial.params['class_weight']
}

best_pipeline = full_pipeline.set_params(**best_params)
#%% md
# ## Evalution Best Model
#%%
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)
test_score = accuracy_score(y_test, y_pred)