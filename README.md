# AML_Pr1
Project 1 for the course of Advanced Machine Learning. The goal is to predict a person’s age from manually perturbed brain image data. Evaluation is based on R2_score.

# RESULTS
33th team out of 144 teams.
Score: 0.74 (1st team score: 0.77. Last team score: 0.25).

#REPORT
Preprocessing:
- the missing valued were filled using KNN inputer (and for each column, a new boolean column was created with value 1 if there used to be a nan value, 0 otherwise)
- the outlier detection was implemented through IsolationForest
- the feature reduction system was implemented in the following way:
   - the first 166 features are selected as the top 10% using f_regression through to the function SelectPercentile
   - select the best 40 features using f_regression through to the function SelectKBest
   - the remaining 126 features are compressed in 60 using Autoencoder (implemented with a dropout layer with p=0.2 to avoid overfitting)
- the dataset is scaled through StandardScaler

Regression:
- stacking of 4 regressors:
   - GaussianProcessRegressor (with RationalQuadratic as kernel)
   - ExtraTreesRegressor (with 5000 estimators)
   - Support Vector Machine Regressor (with gridsearch to find the best hyperparameters and RBF kernel)
   - GradientBoostingRegressor with 80 estimators

Since the autoencoding has a pseudo-random component, the whole process was executed several times and submitted an average of the result.
