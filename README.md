# Clothing Reviews Analysis and Recommendation Prediction Pipeline

This project implements a machine learning pipeline to predict whether customers recommend clothing items based on their reviews. The pipeline handles multiple types of data including numerical, categorical, and text features.

## Project Overview

The goal is to create a robust machine learning pipeline that can predict product recommendations based on customer reviews and other features. The pipeline processes different types of data and combines them into a unified model.

## Features

The dataset includes the following features:

- **Numerical Features**:
  - Age: Reviewer's age
  - Positive Feedback Count: Number of other customers who found the review positive

- **Categorical Features**:
  - Division Name: High-level product division
  - Department Name: Product department name
  - Class Name: Product class name

- **Text Features**:
  - Title: Review title
  - Review Text: Main review content

- **Target Variable**:
  - Recommended IND: Binary variable (1 = recommended, 0 = not recommended)

## Project Structure

```
.
├── data/
│   └── reviews.csv         # Dataset file
├── starter/
│   ├── starter.py         # Main implementation file
│   └── starter.ipynb      # Jupyter notebook version
└── README.md              # Project documentation
```

## Implementation Details

### Data Processing
- Handles missing values
- Processes numerical, categorical, and text features
- Implements custom text transformers for feature extraction

### Feature Engineering
- Numerical features: Standard scaling
- Categorical features: One-hot encoding
- Text features: Custom text transformer with statistical features

### Model Pipeline
- Combines preprocessing steps for different feature types
- Implements RandomForestClassifier with optimized hyperparameters
- Uses Optuna for hyperparameter optimization

### Performance Optimization
- Parallel processing for text analysis
- Optimized CPU utilization
- Efficient memory management

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
python starter/starter.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook starter/starter.ipynb
```

## Model Evaluation

The pipeline includes comprehensive evaluation metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Performance Optimization

The implementation includes several optimizations:
- Parallel processing for text analysis
- Efficient memory management
- Optimized CPU utilization
- Early stopping for model training

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- optuna
- joblib

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
