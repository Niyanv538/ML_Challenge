#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import logging
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


# In[4]:


train = pd.DataFrame.from_records(json.load(open('train.json'))) 
test = pd.DataFrame.from_records(json.load(open('test.json')))


# In[11]:


# Analyze the data by first observing the first 10 rows of the training data
print(train.head(10)) 


# In[18]:


# Check the type of the references column for better analysis
print(type(train['references'][0]))


# In[13]:


# Analyze the dataset further- shape and the column names and data types of the full training dataset
print("Train shape:", train.shape)

print("Columns and data types in train data:")
print(train.info())

# Check for any missing values
print("Missing values in the train data:")
print(train.isnull().sum())


# In[15]:


# Also analyze the test data
print("Test shape:", test.shape)
print("Columns and data types in test data:")
print(test.info())


# In[17]:


# Check the first few rows
print(test.head())
# Get basic statistics for numerical columns
print(test.describe())


# In[7]:


# Check for missing values in each column
print(test.isnull().sum())
# Rows with any missing values
print(test[test.isnull().any(axis=1)])


# In[9]:


# Checking if the ID columns are empty 
print('ID' in test.columns)


# In[19]:


# Visualization analysis of the training dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Check distribution of the target variable = n_citations
sns.histplot(train['n_citation'], kde=True, bins=30)
plt.title("Distribution of n_citation")
plt.xlabel("Number of Citations")
plt.ylabel("Frequency")
plt.show()

# It seems like the training data is skewed. Most papers do not have that many citations 


# In[23]:


# Statistical summary of the target variable => n_citation
summary_stats = train['n_citation'].describe(percentiles=[0.25, 0.5, 0.75])
print("Summary Statistics for n_citation:")
print(summary_stats)

# indicating a typical paper ~38.62 citations.A standard deviation of 125.94 suggests significant variability in number of citations.
# The most cited paper has 25,835 citations, which is an influential and/or a significant outlier 
# min = 0, 0There are papers with 0 citations


# In[27]:


# Applying a Log transformation to interpret the skewness
train['log_n_citation'] = np.log1p(train['n_citation']) 

# Plotting the log-transformed n_citation distribution
plt.figure(figsize=(4, 3))  # Adjusted smaller size for my display preference 
sns.histplot(train['log_n_citation'], kde=True, bins=30)
plt.title("Log-Transformed Distribution of n_citation", fontsize=10)
plt.xlabel("Log(Number of Citations)", fontsize=8)
plt.ylabel("Frequency", fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()


# In[31]:


# Plotting the distribution of the year to publications

sns.histplot(train['year'], kde=False, bins=30)
plt.title("Distribution of Publication Years")
plt.xlabel("Year of Publication")
plt.ylabel("Number of Papers")
plt.show()

# It seems that more papers are published further down the line of the years, with an increase starting from 1980 
# and a rapid increase towards 2020, with a peak between 2010-2020 


# In[35]:


# Scatter plot analysis: The Year vs. Number of Citations

# Calculating the correlation between year of publication and number of citations 
correlation = train['year'].corr(train['n_citation'])
print(f"Correlation between year of publication and number of citations: {correlation:.2f}")

sns.scatterplot(data=train, x='year', y='n_citation')
plt.title("Year of Publication vs. Number of Citations")
plt.xlabel("Year")
plt.ylabel("Number of Citations")
plt.show()

# Weak Negative Correlation (-0.12). 
# older papers tend to have more citations as per this plot, but the relationship is weak, contradicting the previous assumptions.
# This suggests that while publication year (paper age) might play a role in citations, other factors (e.g., venue, quality of research, authorship) might have more influence.
# Newer papers have more citations than old papers. A few papers have over 25,000 citations and could be interpreted as either very influential papers or outliers.
# Such extreme data-points (outliers) can disproportionately affect model predictions, thus we shall handle them carefully during analysis (e.g., by using log transformation).

# However, as per the analysis of the Year of Publication distribution, this increase in citations could be due to the rapid increase in publications


# In[43]:


# Box plot analysis of categorical features:'venue' vs. 'n_citation'
top_venues = train['venue'].value_counts().head(10).index
plt.figure(figsize=(6, 4))  # Adjust the width and height here
sns.boxplot(data=train[train['venue'].isin(top_venues)], x='venue', y='n_citation')
plt.title("Citations by Venues", fontsize=10)
plt.xlabel("Venue", fontsize=8)
plt.ylabel("Number of Citations", fontsize=10)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# Venues such as "Language processing" or "Computer Science" tend to have more influential papers with more citations compared to others


# In[49]:


# Check for the values in the empty 'venue' column
print(train['venue'].value_counts())

# Filtering the rows corresponding to the "empty" third variable to check it
empty_venue = train[train['venue'] == 'third_variable_name']
print(empty_venue)


# In[53]:


print("Missing values for the venue feature:")
print(train['venue'].isnull().sum())  # no venue has missing data, so the empty data in the venue column might indicate that it is empty
# string or has indeed 0 citations


# In[39]:


# Calculate the total number of citations per venue
venue_citations = train.groupby('venue')['n_citation'].sum().sort_values(ascending=False)

# Display the total number of citations per venue
print("Number of Citations per Venue:")
print(venue_citations)


# In[57]:


# Manually applying feature engineering to the textual features abstract and title by making it numerical - length,
# in order to visualize its correlation to the n_citation feature and improve interpretability in the dataset,
# as well as when running the model.

# For training data
train['abstract_length'] = train['abstract'].str.len()  # Abstract length
train['title_length'] = train['title'].str.len()  # Title length

# Analyzing the correlations in the training data
print("Correlation with n_citation:")
print(train[['abstract_length', 'title_length', 'n_citation']].corr())

sns.scatterplot(data=train, x='abstract_length', y='n_citation')
plt.title("Abstract Length vs. Number of Citations")
plt.xlabel("Abstract Length (Characters)")
plt.ylabel("Number of Citations")
plt.show()

sns.scatterplot(data=train, x='title_length', y='n_citation')
plt.title("Title Length vs. Number of Citations")
plt.xlabel("Title Length (Characters)")
plt.ylabel("Number of Citations")
plt.show()

# Applying the engineered features also for test data so they match and the model does not throw an error
test['abstract_length'] = test['abstract'].str.len()  # Abstract length
test['title_length'] = test['title'].str.len()  # Title length


# In[69]:


# Skewness. A shorter abstract and title result in more citations. The relationships of abstract and title length 
# with the target variable are low, which might call for models that don't require correlated variables. 


# In[71]:


# Based on the data characteristics (n_citation is skewed, features like abstract and authors are complex and also skewed) 
# non-linear models are likely better suited


# In[6]:


# feature engineering for all the features
def create_engineered_features(data, train_data=None):
    # the interaction term or abstract length and title length for more correlation with n_citations and get a lower MAE
    data['abstract_length'] = data['abstract'].str.len()
    data['title_length'] = data['title'].str.len()
    data['title_abstract_product'] = data['abstract_length'] * data['title_length']
    data['title_abstract_ratio'] = data['title_length'] / (data['abstract_length'] + 1)  # Avoid division by zero
    
    # Target Encoding applied to both data sets 
    if 'n_citation' in data.columns:  # For train data
        data['venue_encoded'] = target_encode(data, col='venue', target='n_citation')
    elif train_data is not None:  # For test data
        global_means = train_data.groupby('venue')['n_citation'].mean()
        data['venue_encoded'] = data['venue'].map(global_means).fillna(train_data['n_citation'].mean())
    else:
        raise ValueError("Training data must be provided for target encoding on test data.")
    
    # Engineering the references feature
    data = engineer_references(data, train_data)
    
    # Adding interaction terms so the model can better capture non-linear transformations
    data['interaction_abstract_title'] = data['abstract_length'] * data['title_length'] # logarithmic transformations to handle the skewed data
    data['log_title_length'] = np.log1p(data['title_length'])
    data['log_abstract_length'] = np.log1p(data['abstract_length']) 
    
    return data


# In[9]:


# Configuring level logging throughout the analysis for interpretability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Feature Engineering Functions for the main function 
def target_encode(train, col, target, n_splits=5): # Using target encoding to capture a predictive relationship of the variable to n_citation
    # and converts the categorical feature into a single column of the means for better computational time and meaningfulness to the target
    """
    Perform target encoding for a categorical feature using KFold cross-validation.
    """
    from sklearn.model_selection import KFold  # Using KFold cross-validation reduces the risk of overfitting and data leakage

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    encoded_values = np.zeros(len(train))

    for train_idx, val_idx in kf.split(train):
        train_fold, val_fold = train.iloc[train_idx], train.iloc[val_idx]
        means = train_fold.groupby(col)[target].mean()
        encoded_values[val_idx] = val_fold[col].map(means)

    # Using the mean for unseen categories and to increase robustness when training for this categorical feature
    global_mean = train[target].mean()
    encoded_values = np.where(np.isnan(encoded_values), global_mean, encoded_values)

    return encoded_values

# Transforming the listwise textual references to a binary numerical feature. Papers that cite "prominent" or highly-cited works are
# correlated to the number of citations -> [1] 'prominent' works, [0] non-prominent
def engineer_references(data, train_data=None):
    """
    Add a binary feature indicating whether references include prominent works.
    """
    if train_data is None:  # For training data
        global top_references
        reference_counts = data['references'].explode().value_counts()
        top_references = set(reference_counts.head(100).index)
    elif 'top_references' not in globals():
        raise ValueError("Prominent references must be computed during training.")

    # Creating the binary feature
    data['prominent_reference'] = data['references'].apply(
        lambda refs: any(ref in top_references for ref in refs) if isinstance(refs, list) else False
    ).astype(int)

    return data

# 
def create_engineered_features(data, train_data=None):
    """
    Perform feature engineering on the dataset, including text features and target encoding.
    """
    # Transforming the Textual Features into numerical (length) for regression.
    data['abstract_length'] = data['abstract'].str.len()
    data['title_length'] = data['title'].str.len()
    data['title_abstract_product'] = data['abstract_length'] * data['title_length'] # interaction for more complex relationships. 
    data['title_abstract_ratio'] = data['title_length'] / (data['abstract_length'] + 1)  # ratio reveals relative differences between title and abstract lengths show the
    # Avoid division by zero to avoid infinite values 

    # Target Encoding
    if 'n_citation' in data.columns:  # For train data
        data['venue_encoded'] = target_encode(data, col='venue', target='n_citation')
    elif train_data is not None:  # For validation/test data
        global_means = train_data.groupby('venue')['n_citation'].mean()
        data['venue_encoded'] = data['venue'].map(global_means).fillna(train_data['n_citation'].mean())
    else:
        raise ValueError("Training data must be provided for target encoding on test data.")

    # Engineer References Feature
    data = engineer_references(data, train_data)

    # Adding Interaction Terms and Logarithmic Transformations for handling skewness 
    data['interaction_abstract_title'] = data['abstract_length'] * data['title_length']
    data['log_title_length'] = np.log1p(data['title_length'])
    data['log_abstract_length'] = np.log1p(data['abstract_length'])

    return data


# Main Function
def main():
    # Load train and test datasets
    logging.info("Loading datasets...")
    train = pd.DataFrame.from_records(json.load(open('train.json')))
    test = pd.DataFrame.from_records(json.load(open('test.json')))


    # Splitting the training dataset into training and validation subsets
    logging.info("Splitting data into training and validation sets...")
    train, validation = train_test_split(train, test_size=1/3, random_state=123)

    # Applying feature engineering to each subset
    logging.info("Applying feature engineering to training data...")
    train = create_engineered_features(train)

    logging.info("Applying feature engineering to validation data...")
    validation = create_engineered_features(validation, train_data=train)

    logging.info("Applying feature engineering to test data...")
    test = create_engineered_features(test, train_data=train)

    # Log transformation
    logging.info("Applying log transformation to the target variable...")
    train['n_citation'] = np.log1p(train['n_citation'])
    validation['n_citation'] = np.log1p(validation['n_citation'])

    # Normalize Numerical Features & apply to each subset
    logging.info("Normalizing numerical features...")
    numerical_cols = ['abstract_length', 'title_length', 'title_abstract_product', 'title_abstract_ratio',
                      'venue_encoded', 'year', 'interaction_abstract_title', 'log_title_length', 'log_abstract_length']
    scaler = StandardScaler()
    train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
    validation[numerical_cols] = scaler.transform(validation[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])

    # Remove low-variance features & apply to each subset
    logging.info("Removing low-variance features...")
    low_variance_cols = train[numerical_cols].var()
    low_variance_cols = low_variance_cols[low_variance_cols < 1e-5].index
    train.drop(columns=low_variance_cols, inplace=True)
    validation.drop(columns=low_variance_cols, inplace=True)
    test.drop(columns=low_variance_cols, inplace=True)

    # Update feature columns after removing low-variance features
    feature_cols = [col for col in numerical_cols if col not in low_variance_cols] + ['prominent_reference']

    # Defining my LightGBM model with Randomized Search for hyperparameter tuning
    param_distributions = {
    'n_estimators': [500, 1000, 1500],  # More trees to balance lower learning rates
    'learning_rate': [0.01, 0.03, 0.05],  # Lower learning rate for stability & convergence for no overfitting
    'max_depth': [5, 6, 7],  # Restrict depth to prevent overfitting
    'num_leaves': [31, 50],  # Standard leaf sizes
    'colsample_bytree': [0.7, 0.8],  # Feature subsampling
    'min_child_samples': [20, 50, 100],  # Avoid splits on small noisy subsets
    'reg_alpha': [0.1, 0.5, 1.0],  # L1 regularization to simplify model
    'reg_lambda': [0.1, 0.5, 1.0],  # L2 regularization for stability
}

# using a gradient boosting model version = lightGBM suitable for larger datasets with faster computation and memory efficiency, can handle 
# imbalanced data as this, non-linear relationships, and is good for various types of features, even though I converted mine to numerical for easier interpretation
# Using randomizedSearchCV for fixed tuning for computational space, flexibility in tuning, and suitable for larger datasets like this one

    random_search = RandomizedSearchCV(
        estimator=LGBMRegressor(random_state=123),
        param_distributions=param_distributions,
        n_iter=20, 
        scoring='neg_mean_absolute_error',
        cv=3, # generalizability and enough folds for a good performance and good computational time
        verbose=2,
        random_state=123,
        n_jobs=-1 # entire memory usage
    )

    label = 'n_citation'

    # logging
    logging.info("Performing Randomized Search for LightGBM...")
    random_search.fit(train[feature_cols], train[label])
    logging.info(f"Best Parameters for LightGBM: {random_search.best_params_}")
    logging.info(f"Best MAE from Randomized Search: {-random_search.best_score_:.2f}")

    # Evaluating the best LightGBM model
    best_lgbm = random_search.best_estimator_
    for split_name, split in [("train     ", train), ("validation", validation)]:
        pred = np.expm1(best_lgbm.predict(split[feature_cols]))
        mae = mean_absolute_error(np.expm1(split[label]), pred)  # Convert back from log scale
        logging.info(f"LightGBM {split_name} MAE: {mae:.2f}")

    # Prediction on test set & saving the predictions
    logging.info("Making predictions on the test set...")
    predicted = np.expm1(best_lgbm.predict(test[feature_cols]))
    test['n_citation'] = predicted
    json.dump(test[['n_citation']].to_dict(orient='records'), open('predicted.json', 'w'), indent=2)
    logging.info("Predictions saved to predicted.json")

# Run the main function
main()


# In[ ]:


# I tried everything to get rid of the warnings '[Warning] No further splits with positive gain, best gain: -inf' which probably mean 
# exhausted splits or restricted or restricted paramters

