Using a gradient boosting model version, lightGBM, to predict number of paper citations by lowering the MAE value of a baseline model (ridge) already used for prediction = 33.0 as baseline. This model is suitable for larger datasets with faster computation and memory efficiency that can handle 
imbalanced data as this, non-linear relationships, and is good for various types of features. RandomizedCVSearch was used to hyperparameter tuning for computational space, flexibility in tuning, and suitable for larger datasets

This project conducted data Splitting, Feature engineering by removing low variance features and normalizing numerical features. My Light GBM attempt lowered the MAE to 29, ranking as top 15 in the course class competition board.
