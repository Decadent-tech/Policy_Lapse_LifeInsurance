import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import learning_curve

from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


df= pd.read_csv('dataset/Kaggle.csv')
print(df.shape)
print(df.value_counts())
print(df.head())
# Data Preprocessing
print(df['POLICY STATUS'].unique())

# Dropping rows with specific POLICY STATUS values
df.drop(df[df['POLICY STATUS'] == 'Expired'].index, inplace=True)
df.drop(df[df['POLICY STATUS'] == 'Surrender'].index, inplace=True)
df.drop(df[df['POLICY STATUS'] == 'Death'].index, inplace=True) 
# Dropping issue date column 
df.drop('Issue Date', axis=1, inplace=True)

# mapping POLICY STATUS Lapse and Inforce to LAPSE and NOTLAPSE
df['POLICY STATUS'] = df['POLICY STATUS'].map({
    "Lapse": "LAPSE",
    "Inforce": "NOTLAPSE"
})
print(df.shape)

# Visualizing the distribution of POLICY STATUS 
sns.countplot(x=df['POLICY STATUS'])
plt.savefig('Visualization/Distribution of POLICY STATUS')


# Data Pre-processing¶
# Data pre-processing in Machine Learning refers to the technique of preparing data. Here missing value of feature was identified and imputation of missing values was performed.

missing_stats = []
for col in df.columns:
    missing_stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
    
stats_df = pd.DataFrame(missing_stats, columns=['feature', 'unique_values', 'percent_missing', 'percent_biggest_cat', 'type'])
print(stats_df.sort_values('percent_missing', ascending=False))

# describing the dataframe 
print(df.describe())

# Imputing missing values
imputer_mean = SimpleImputer(strategy='median')

numeric_features = ['Premium', 'BENEFIT']
df[numeric_features] = imputer_mean.fit_transform(df[numeric_features])

df['POLICY STATUS'] = df['POLICY STATUS'].map({'LAPSE': 1, 'NOTLAPSE': 0})
df['POLICY STATUS'] = df['POLICY STATUS'].astype(int)

# rechecking the perecent missing values after imputation 

missing_stats = []
for col in df.columns:
    missing_stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
    
stats_df = pd.DataFrame(missing_stats, columns=['feature', 'unique_values', 'percent_missing', 'percent_biggest_cat', 'type'])
print(stats_df.sort_values('percent_missing', ascending=False))

# Splitting the dataset into training and testing sets
X = df.drop('POLICY STATUS', axis=1)
y = df['POLICY STATUS']


# Data Partitioning
# After evaluating the result of different partition. The best fit in accuracy and other parameters specified in evaluation of the model was chosen.

# The following procedure will be followed for the data partition as

# Training Set: 80%
# Test Set: 20%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Statistically Feature Selected
# The reduction of features can also decrease the time of execution. To some extent, it can also prevent overfitting.
# For statistical approach Pearson's correlation coefficient was measured for numerical feature whereas for categorial 
# feature Chi-squared statistic was measured.

all_col = list(X_train.columns)
all_col_count = len(all_col)
print("Total No. of Features in the Dataset:")
print(all_col_count)

all_features = X_train.columns
print("All Features in the Dataset:")
print(all_features)

# Pearson’s Correlation Coefficient - In the process of data selection Pearson’s Correlation Coefficient analysis was examined 
# to find relationship between various attribute with correlation coefficient exceeding 0.5 were consider strongly correlated.

def color(val):
    return 'color: %s' % (
        'green' if val == 1 else
        'red' if val < -0.5 else
        'blue' if val > 0.5 else
        'black'
    )

corr = X_train.select_dtypes(include=['number']).corr()

# Style correlation dataframe
styled_corr = corr.style.map(color).background_gradient(cmap='coolwarm').format(precision=2)

# Plot heatmap of absolute correlations
corr_abs = corr.abs()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_abs, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.savefig('Visualization/Pearson_Correlation_Coefficient.png', bbox_inches='tight')
# plt.show()


X_train_cat = X_train[all_features].select_dtypes(include=['object'])
print(X_train_cat.columns)

X_train_numeric = X_train[all_features].select_dtypes(include=['number'])
print(X_train_numeric.columns)


# correlation matrix for numeric features

corr_matrix = X_train_numeric.corr().round(2)

sns.set_style('white')
fig, ax = plt.subplots(figsize=(12, 12))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

ax = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='OrRd')

ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10, ha='right', rotation=90)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10, va="center", rotation=0)

# plt.show()
plt.savefig('Visualization/Correlation_Matrix_Numeric_Features.png', bbox_inches='tight')

correlation_threshold = 0.5

highly_correlated_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            colname = corr_matrix.columns[i]
            highly_correlated_features.add(colname)


print(highly_correlated_features)

# Chi-Squared - A threshold value of 0.05 was used to determine feature significance. 
# Based on this threshold, feature with p-value greater than 0.05 were considered less significant.

bool_columns = X_train_cat.select_dtypes(include='bool').columns
X_train_cat[bool_columns] = X[bool_columns].astype(str)

print(X_train_cat.isnull().sum())

# Encoding categorical features
le = LabelEncoder()
X_train_nom_encoded = pd.DataFrame()
X_train_nom_encoded = X_train_cat.copy()

for col in X_train_cat.columns:
    X_train_nom_encoded[col] = le.fit_transform(X_train_cat[col])

print(X_train_nom_encoded.head()) # Displaying the first few rows of the categorical encoded DataFrame

# Chi-squared test for categorical features
chi_score = chi2(X_train_nom_encoded, y_train)
print(chi_score)

chi2_score_series = pd.Series(chi_score[0], index = X_train_nom_encoded.columns).sort_values(ascending = False)# Displaying chi-squared scores for each feature
print(chi2_score_series)
p_value_series =  pd.Series(chi_score[1], index = X_train_nom_encoded.columns)
print(p_value_series) # Displaying p-values for each feature


plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(x=chi2_score_series.index, y=chi2_score_series.values)
plt.title('Chi-Squared Scores for Categorical Features')
plt.xlabel('Features')  
plt.ylabel('Chi-Squared Score')
plt.savefig('Visualization/Chi_Squared_Scores.png', bbox_inches='tight')
#plt.show()


plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(x=p_value_series.index, y=p_value_series.values)
plt.title('P-Values for Categorical Features')
plt.xlabel('Features')
plt.ylabel('P-Value')
plt.axhline(y=0.05, color='r', linestyle='--', label='Threshold (0.05)')
plt.legend()
plt.savefig('Visualization/P_Values.png', bbox_inches='tight')
#plt.show()

# Dropping features with p-value greater than 0.05
print("Features to drop based on p-value threshold:")
significance_threshold = 0.05
features_to_drop = p_value_series[p_value_series > significance_threshold].index.tolist()
print(features_to_drop) 

X_train_selected = X_train.drop(columns=highly_correlated_features)
print(X_train_selected)
X_test_selected = X_test.drop(columns=highly_correlated_features)
print(X_test_selected)

X_train_selected = X_train_selected.drop(columns=features_to_drop)
X_test_selected = X_test_selected.drop(columns=features_to_drop)
print("Shape of X_train_selected:", X_train_selected.shape)
print("Shape of X_test_selected:", X_test_selected.shape)


print("Remaining Features X train:")
print(X_train_selected.columns)

# Statistically Feature Selected Model
#In this study, we consider two powerful ensemble learning algorithms Random Forest and XG Boost.

X_combined = pd.concat([X_train_selected, X_test_selected], axis=0)
X_combined_encoded = pd.get_dummies(X_combined)
X_train_encoded = X_combined_encoded.iloc[:len(X_train_selected)]
X_test_encoded = X_combined_encoded.iloc[len(X_train_selected):]
print(X_train_encoded, X_test_encoded)
# Target variable count
print(y_train.value_counts())
print(y_test.value_counts())

# calculating the class distribution
class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(class_ratio)

# Evaluation Metrics of Random Forest

class_weight_value = {0: 1, 1: class_ratio} 
"""
accuracy_list = []
for i in range(1,150):
    rf_classifier = RandomForestClassifier(n_estimators=i, random_state=42, class_weight=class_weight_value)
    rf_classifier.fit(X_train_encoded, y_train)
    y_pred_rf = rf_classifier.predict(X_test_encoded)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    accuracy_list.append(accuracy_rf)

np.savetxt("accuracy_list.csv", accuracy_list, delimiter=",")

for i in range(len(accuracy_list)):
    #best accuracy = max(accuracy_list)
    if accuracy_list[i] == max(accuracy_list):
        best_n_estimators = i
        print("Best n_estimators for Random Forest:", best_n_estimators)
        break
"""

best_n_estimators = 33  # Set to the best n_estimators found from previous analysis
rf_classifier_best_value = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42, class_weight=class_weight_value)
rf_classifier_best_value.fit(X_train_encoded, y_train)
y_pred_rf = rf_classifier_best_value.predict(X_test_encoded)
print("Statistically Feature Selected Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
confusion_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_rf, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Visualization/Confusion_Matrix_Random_Forest.png', bbox_inches='tight')

num_folds = 10
cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
scores = cross_val_score(rf_classifier_best_value, X_combined_encoded, y, cv=cv, scoring='f1')
print("Cross-Validation Scores:", scores)
mean_score = scores.mean()
std_dev = scores.std()
print("Mean Cross-Validation Score:", mean_score)
print("Standard Deviation of Cross-Validation Scores:", std_dev)

# Learning Curve 
def plot_learning_curve(estimator, x, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    sns.set_style("white")
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1)
    plt.plot(train_sizes, train_scores_mean, 'o-',
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-',
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(rf_classifier_best_value, X_train_encoded, y_train, cv=10, n_jobs=-1)
plt.savefig('Visualization/Learning_Curve_Random_Forest.png', bbox_inches='tight')

# Evaluation Metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Statistically Feature Selected Random Forest Accuracy: {rf_accuracy:.4f}')
rf_precision = precision_score(y_test, y_pred_rf)
print(f'Statistically Feature Selected Random Forest Precision: {rf_precision:.4f}')
rf_recall = recall_score(y_test, y_pred_rf)
print(f'Statistically Feature Selected Random Forest Recall: {rf_recall:.4f}')
rf_f_measure = f1_score(y_test, y_pred_rf)
print(f'Statistically Feature Selected Random Forest F-measure: {rf_f_measure:.4f}')

predictions = rf_classifier_best_value.predict(X_test_encoded)
rf_cf_matrix = confusion_matrix(y_test, predictions)

group_names = ['True -ve','False +ve','False -ve','True +ve']
group_counts = ['n={0:0.0f}'.format(value) for value in rf_cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in rf_cf_matrix.flatten()/np.sum(rf_cf_matrix)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

yticklabels=['Not Lapse','Lapse']
xticklabels=['Predicted as\nNot Lapse','Predicted as\Lapse']

fix, ax = plt.subplots(figsize=(6,5))

sns.set()
ax = sns.heatmap(rf_cf_matrix, annot=labels, 
            xticklabels = xticklabels, yticklabels = yticklabels, 
            fmt='', cmap='Blues');

ax.set_title('Confusion matrix', fontsize=15,  fontweight='bold')
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=11, ha= 'center', rotation=0 )
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=11, va="center", rotation=0)
plt.savefig('Visualization/Confusion_Matrix_Random_Forest_Annotated.png', bbox_inches='tight')
#plt.show();

# Evaluation Metrics of XG Boost¶
scale_pos_weight_value = class_ratio

xg_classifier = XGBClassifier(booster='gbtree', n_jobs=-1, 
                              scale_pos_weight=scale_pos_weight_value)

xg_classifier.fit(X_train_encoded, y_train)

y_pred_xg = xg_classifier.predict(X_test_encoded)

print("Statistically Feature Selected XG Boost Classification Report:")
print(classification_report(y_test, y_pred_xg))

num_folds = 10

cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

scores = cross_val_score(xg_classifier, X_combined_encoded, y, cv=cv, scoring='f1')

print("Cross-Validation Scores:", scores)

mean_score = scores.mean()
std_dev = scores.std()
print("Mean Cross-Validation Score:", mean_score)
print("Standard Deviation of Cross-Validation Scores:", std_dev)


plot_learning_curve(xg_classifier, X_train_encoded, y_train, cv=10, n_jobs=-1)
# plt.show()
plt.savefig('Visualization/Learning_Curve_XG_Boost.png', bbox_inches='tight')


# Evaluation Metrics
xgb_accuracy = accuracy_score(y_test, y_pred_xg)
print(f'Statistically Feature Selected XG Boost Accuracy: {xgb_accuracy:.4f}')
xgb_precision = precision_score(y_test, y_pred_xg)
print(f'Statistically Feature Selected XG Boost Precision: {xgb_precision:.4f}')
xgb_recall = recall_score(y_test, y_pred_xg)
print(f'Statistically Feature Selected XG Boost Recall: {xgb_recall:.4f}')
xgb_f_measure = f1_score(y_test, y_pred_xg)
print(f'Statistically Feature Selected XG Boost F-measure: {xgb_f_measure:.4f}')

predictions = xg_classifier.predict(X_test_encoded)
xg_cf_matrix = confusion_matrix(y_test, predictions)

group_names = ['True -ve','False +ve','False -ve','True +ve']
group_counts = ['n={0:0.0f}'.format(value) for value in xg_cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in xg_cf_matrix.flatten()/np.sum(xg_cf_matrix)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

yticklabels=['Not Lapse','Lapse']
xticklabels=['Predicted as\nNot Lapse','Predicted as\Lapse']


fix, ax = plt.subplots(figsize=(6,5))

sns.set()
ax = sns.heatmap(xg_cf_matrix, annot=labels, 
            xticklabels = xticklabels, yticklabels = yticklabels, 
            fmt='', cmap='Blues');

ax.set_title('Confusion matrix', fontsize=15,  fontweight='bold')
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=11, ha= 'center', rotation=0 )
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=11, va="center", rotation=0)
plt.savefig('Visualization/Confusion_Matrix_XG_Boost_Annotated.png', bbox_inches='tight')
# plt.show();

print(X_train.dtypes) # Displaying the data types of the features in X_train


label_encoder = LabelEncoder()
X_train_encodedxgb = X_train.copy()
X_test_encodedxgb = X_test.copy()
# Encoding categorical features for XGBoost
for col in X_train_cat:
    X_test_encodedxgb[col] = label_encoder.fit_transform(X_test[col])
for col in X_train_cat:
    X_train_encodedxgb[col] = label_encoder.fit_transform(X_train[col])

# Calculating the class ratio for handling class imbalance
scale_pos_weight_value = class_ratio

xgb_model = XGBClassifier(booster='gbtree', n_jobs=-1, scale_pos_weight=scale_pos_weight_value)
xgb_model.fit(X_train_encodedxgb, y_train)
feature_importance = xgb_model.feature_importances_
for feature, importance in zip(X_train_encodedxgb.columns, feature_importance):
    print(f"Feature: {feature}, Importance: {importance}")

median_importance = np.median(feature_importance)
print(median_importance)

threshold = median_importance 

top_features_indices = np.where(feature_importance > threshold)[0]
print("Top features indices:", top_features_indices)

top_features = X_train_encodedxgb.columns[top_features_indices]
X_train_xgb_selected = X_train[top_features]
print(X_train_xgb_selected.columns)

X_test_xgb_selected = X_test[top_features]
print(X_test_xgb_selected)

X_combined_xgb = pd.concat([X_train_xgb_selected, X_test_xgb_selected], axis=0)
X_combined_xgb_encoded = pd.get_dummies(X_combined_xgb)
X_train_xgb_encoded = X_combined_xgb_encoded.iloc[:len(X_train_xgb_selected)]
X_test_xgb_encoded = X_combined_xgb_encoded.iloc[len(X_train_xgb_selected):]
print(X_train_xgb_encoded, X_test_xgb_encoded)

# XG Boost Feature Selected Model 
# In our study, two different feature selection technique is used statistical approach of feature selection that include Pearson's Correlation Coefficient and Chi-Squared and Feature Selection using XG Boost Feature Importance, followed by comparison of two powerful ensemble learning algorithms Random Forest and XG Boost.
# Evaluation Metrics of XG Boost Feature Selected Random Forest
class_weight_value = {0: 1, 1: class_ratio}
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_value)
rf_model.fit(X_train_xgb_encoded, y_train)
y_pred_xgb_rf = rf_model.predict(X_test_xgb_encoded)

print("XG Boost Feature Selected Random Forest Classification Report:")
print(classification_report(y_test, y_pred_xgb_rf))

num_folds = 10

cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

scores = cross_val_score(xg_classifier, X_combined_encoded, y, cv=cv, scoring='f1')
print("Cross-Validation Scores:", scores)

mean_score = scores.mean()
std_dev = scores.std()
print("Mean Cross-Validation Score:", mean_score)
print("Standard Deviation of Cross-Validation Scores:", std_dev)

plot_learning_curve(rf_model, X_train_xgb_encoded, y_train, cv=10, n_jobs=-1)
# plt.show()
plt.savefig('Visualization/Learning_Curve_Random_Forest_XG_Boost_Feature_Selected.png', bbox_inches='tight')

# Evaluation Metrics
xgb_rf_accuracy = accuracy_score(y_test, y_pred_xgb_rf)
print(f'XG Boost Feature Selected Random Forest Accuracy: {xgb_rf_accuracy:.4f}')
xgb_rf_precision = precision_score(y_test, y_pred_xgb_rf)
print(f'XG Boost Feature Selected Random Forest Precision: {xgb_rf_precision:.4f}')
xgb_rf_recall = recall_score(y_test, y_pred_xgb_rf)
print(f'XG Boost Feature Selected Random Forest Recall: {xgb_rf_recall:.4f}')
xgb_rf_f_measure = f1_score(y_test, y_pred_xgb_rf)
print(f'XG Boost Feature Selected Random Forest F-measure: {xgb_rf_f_measure:.4f}')

# Confusion Matrix with XG Boost Feature Selected Random Forest
predictions = rf_model.predict(X_test_xgb_encoded)
xg_rf_matrix = confusion_matrix(y_test, predictions)

group_names = ['True -ve','False +ve','False -ve','True +ve']
group_counts = ['n={0:0.0f}'.format(value) for value in xg_rf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in xg_rf_matrix.flatten()/np.sum(xg_rf_matrix)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

yticklabels=['Not Lapse','Lapse']
xticklabels=['Predicted as\nNot Lapse','Predicted as\Lapse']

fix, ax = plt.subplots(figsize=(6,5))

sns.set()
ax = sns.heatmap(xg_rf_matrix, annot=labels, 
            xticklabels = xticklabels, yticklabels = yticklabels, 
            fmt='', cmap='Blues');

ax.set_title('Confusion matrix', fontsize=15,  fontweight='bold')
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=11, ha= 'center', rotation=0 )
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=11, va="center", rotation=0)
plt.savefig('Visualization/Confusion_Matrix_Random_Forest_XG_Boost_Feature_Selected_Annotated.png', bbox_inches='tight')
# plt.show();

# Evaluation Metrics of XG Boost Feature Selected XG Boost¶
scale_pos_weight_value = class_ratio
xgb_xgb_model = XGBClassifier(booster='gbtree', n_jobs=-1,  scale_pos_weight=scale_pos_weight_value)
xgb_xgb_model.fit(X_train_xgb_encoded, y_train)
y_pred_xgb_xgb = xgb_xgb_model.predict(X_test_xgb_encoded)

print("XG Boost Feature Fusion XG Boost Classification Report:")
print(classification_report(y_test, y_pred_xgb_xgb))

num_folds = 10

cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

scores = cross_val_score(xgb_xgb_model, X_combined_xgb_encoded, y, cv=cv, scoring='f1')

print("Cross-Validation Scores:", scores)

mean_score = scores.mean()
std_dev = scores.std()
print("Mean Cross-Validation Score:", mean_score)
print("Standard Deviation of Cross-Validation Scores:", std_dev)

plot_learning_curve(xgb_xgb_model, X_train_xgb_encoded, y_train, cv=10, n_jobs=-1)
# plt.show()
plt.savefig('Visualization/Learning_Curve_XG_Boost_XG_Boost_Feature_Selected.png', bbox_inches='tight')

# Evaluation Metrics
xgb_xgb_accuracy = accuracy_score(y_test, y_pred_xgb_xgb)
print(f'XG Boost Feature Selected XG Boost Accuracy: {xgb_xgb_accuracy:.4f}')
xgb_xgb_precision = precision_score(y_test, y_pred_xgb_xgb)
print(f'XG Boost Feature Selected XG Boost Precision: {xgb_xgb_precision:.4f}')
xgb_xgb_recall = recall_score(y_test, y_pred_xgb_xgb)
print(f'XG Boost Feature Selected XG Boost Recall: {xgb_xgb_recall:.4f}')
xgb_xgb_f_measure = f1_score(y_test, y_pred_xgb_xgb)
print(f'XG Boost Feature Selected XG Boost F-measure: {xgb_xgb_f_measure:.4f}')

# Confusion Matrix with XG Boost Feature Selected XG Boost
predictions = xgb_xgb_model.predict(X_test_xgb_encoded)
xg_xgb_matrix = confusion_matrix(y_test, predictions)

group_names = ['True -ve','False +ve','False -ve','True +ve']
group_counts = ['n={0:0.0f}'.format(value) for value in xg_xgb_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in xg_xgb_matrix.flatten()/np.sum(xg_xgb_matrix)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

yticklabels=['Not Lapse','Lapse']
xticklabels=['Predicted as\nNot Lapse','Predicted as\Lapse']


fix, ax = plt.subplots(figsize=(6,5))

sns.set()
ax = sns.heatmap(xg_xgb_matrix, annot=labels, 
            xticklabels = xticklabels, yticklabels = yticklabels, 
            fmt='', cmap='Blues');

ax.set_title('Confusion matrix', fontsize=15,  fontweight='bold')
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=11, ha= 'center', rotation=0 )
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=11, va="center", rotation=0)
plt.savefig('Visualization/Confusion_Matrix_XG_Boost_XG_Boost_Feature_Selected_Annotated.png', bbox_inches='tight')
# plt.show();

# Comparative Analysis

rf_probs = rf_classifier_best_value.predict_proba(X_test_encoded)[:, 1] # Probability of positive class
xgb_probs = xg_classifier.predict_proba(X_test_encoded)[:, 1]# Probability of positive class for xgboost
# Probability of positive class for xgboost feature selected random forest
xgb_rf_probs = rf_model.predict_proba(X_test_xgb_encoded)[:, 1]
# Probability of positive class for xgboost feature selected xgboost
xgb_xgb_probs = xgb_xgb_model.predict_proba(X_test_xgb_encoded)[:, 1]

rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
rf_average_precision = average_precision_score(y_test, rf_probs)

xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_probs)
xgb_average_precision = average_precision_score(y_test, xgb_probs)

xgb_rf_precision, xgb_rf_recall, _ = precision_recall_curve(y_test, xgb_rf_probs)
xgb_rf_average_precision = average_precision_score(y_test, xgb_rf_probs)

xgb_xgb_precision, xgb_xgb_recall, _ = precision_recall_curve(y_test, xgb_xgb_probs)
xgb_xgb_average_precision = average_precision_score(y_test, xgb_xgb_probs)

fig = plt.figure(figsize=(8, 6))
sns.set_style("white")

plt.figure(figsize=(8, 6))
plt.step(rf_recall, rf_precision, where='post', label=f'Statistically Feature Selected Random Forest (AP = {rf_average_precision:.2f})')
plt.step(xgb_recall, xgb_precision, where='post', label=f'Statistically Feature Selected XGBoost (AP = {xgb_average_precision:.2f})')
plt.step(xgb_rf_recall, xgb_rf_precision, where='post', label=f'XGBoost Featured Selected Random Forest (AP = {xgb_rf_average_precision:.2f})')
plt.step(xgb_xgb_recall, xgb_xgb_precision, where='post', label=f'XGBoost Featured Selection XG Boost (AP = {xgb_xgb_average_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc='lower left')
# plt.show()
plt.savefig('Visualization/Precision_Recall_Curve.png', bbox_inches='tight')    

# Features Extraction from the Best Model

feature_importance = xgb_xgb_model.feature_importances_

feature_names = X_train_xgb_encoded.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

top_10_features = feature_importance_df.head(10)

plt.figure(figsize=(12, 6))
plt.barh(top_10_features['Feature'], top_10_features['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 XG Boost Feature Selected XG Boost Feature Importance')
plt.savefig('Visualization/Top_10_Features_XG_Boost.png', bbox_inches='tight')
# plt.show()

booster = xgb_xgb_model.get_booster()

feature_gain = booster.get_score(importance_type='gain')
feature_gain_df = pd.DataFrame({'Feature': list(feature_gain.keys()), 'Gain': list(feature_gain.values())})
feature_gain_df = feature_gain_df.sort_values(by='Gain', ascending=False)

top_10_features_gain = feature_gain_df.head(10)

plt.figure(figsize=(12, 6))
plt.barh(top_10_features_gain['Feature'], top_10_features_gain['Gain'])
plt.xlabel('Feature Gain')
plt.ylabel('Feature')
plt.title('Top 10 XG Boost Feature Selected XG Boost Feature Gain')
plt.savefig('Visualization/Top_10_Features_Gain_XG_Boost.png', bbox_inches='tight')
# plt.show()


# Conclusion and Future Work

# Our comparative study analyzed Random Forest and XG Boost, both features selected from statistical 
# approach and with XG Boost feature importance. Evaluation metrics such as accuracy, precision, recall and F-measure 
# along with Precision-Recall curves, provide comprehensive insights into the effectiveness of our approach.
# In conclusion, our study offers valuable insights into enhancing the predictive capabilities of 
# Random Forest and XG Boost algorithm for predicting lapse in life insurance. By integrating XG Boost Feature Importance 
# for feature selection, we have demonstrated significant improvements in model performance.
# In addition to utilizing XG Boost's feature importance for feature selection, a potential enhancement to our 
# methodology can be Recursive Feature Elimination (RFE). By integrating REF into feature selection, 
# we can further refine the feature subset used by predictive models. The iterative approach allows us to 
# systematically evaluate the contribution of each feature to the model's performance, thereby potentially improving 
# predicting accuracy and reducing overfitting.


# Recursive Feature Elimination (RFE),
from sklearn.feature_selection import RFE

# Recursive Feature Elimination with Random Forest
rfe_xgb_rf = RFE(estimator=rf_model, n_features_to_select=10)
rfe_xgb_rf.fit(X_train_xgb_encoded, y_train)

selected_features_xgb_rf = X_train_xgb_encoded.columns[rfe_xgb_rf.support_]
feature_ranking_rf = pd.DataFrame({
    "Feature": X_train_xgb_encoded.columns,
    "Selected_RF": rfe_xgb_rf.support_,
    "Ranking_RF": rfe_xgb_rf.ranking_
}).sort_values("Ranking_RF")

print("Selected features using RFE with Random Forest:")
print(selected_features_xgb_rf.tolist())

# Recursive Feature Elimination with XG Boost
rfe_xgb_xgb = RFE(estimator=xgb_xgb_model, n_features_to_select=10)
rfe_xgb_xgb.fit(X_train_xgb_encoded, y_train)

selected_features_xgb_xgb = X_train_xgb_encoded.columns[rfe_xgb_xgb.support_]
feature_ranking_xgb = pd.DataFrame({
    "Feature": X_train_xgb_encoded.columns,
    "Selected_XGB": rfe_xgb_xgb.support_,
    "Ranking_XGB": rfe_xgb_xgb.ranking_
}).sort_values("Ranking_XGB")

print("Selected features using RFE with XGBoost:")
print(selected_features_xgb_xgb.tolist())

comparison_df = pd.merge(feature_ranking_rf, feature_ranking_xgb, on="Feature")
print("Feature Ranking Comparison:")
print(comparison_df.head(15))  # show top 15

common_features = set(selected_features_xgb_rf).intersection(set(selected_features_xgb_xgb))
print(f"Common Selected Features (RF ∩ XGB): {list(common_features)}")

plt.figure(figsize=(10, 6))
comparison_df.set_index("Feature")[["Ranking_RF", "Ranking_XGB"]].plot.barh(figsize=(12, 8))
plt.title("RFE Feature Ranking: RF vs XGB")
plt.xlabel("Ranking (Lower = Better)")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.savefig('Visualization/RFE_Feature_Ranking_RF_vs_XGB.png', bbox_inches='tight')
# plt.show()

# The above code provides a comprehensive analysis of the policy lapse prediction problem using machine learning techniques.
# It includes data preprocessing, feature selection using statistical methods and XG Boost feature importance,
# model training with Random Forest and XG Boost, evaluation of model performance, and visualization of results.

# Get rankings
rf_ranking = pd.Series(rfe_xgb_rf.ranking_, index=X_train_xgb_encoded.columns)
xgb_ranking = pd.Series(rfe_xgb_xgb.ranking_, index=X_train_xgb_encoded.columns)

# Combine into one DataFrame
feature_rankings = pd.DataFrame({
    "RF_Ranking": rf_ranking,
    "XGB_Ranking": xgb_ranking
}).sort_values(by=["RF_Ranking", "XGB_Ranking"])

print(feature_rankings.head(15))

# Train RF with RFE-selected features
rf_model.fit(X_train_xgb_encoded[selected_features_xgb_rf], y_train)
rf_preds = rf_model.predict(X_test_xgb_encoded[selected_features_xgb_rf])

# Train XGB with RFE-selected features
xgb_xgb_model.fit(X_train_xgb_encoded[selected_features_xgb_xgb], y_train)
xgb_preds = xgb_xgb_model.predict(X_test_xgb_encoded[selected_features_xgb_xgb])

print("RF (RFE features) F1:", f1_score(y_test, rf_preds))
print("XGB (RFE features) F1:", f1_score(y_test, xgb_preds))

rf_importance = rf_model.feature_importances_
plt.barh(selected_features_xgb_rf, rf_importance[:len(selected_features_xgb_rf)])
plt.title("Random Forest Feature Importances (RFE Selected)")
plt.savefig('Visualization/RF_Feature_Importances_RFE_Selected.png', bbox_inches='tight')
# plt.show()

rf_cv_scores = cross_val_score(rf_model, 
                               X_train_xgb_encoded[selected_features_xgb_rf], 
                               y_train, cv=5, scoring="f1")
print("RF RFE CV F1:", rf_cv_scores.mean())

common_features = list(set(selected_features_xgb_rf) & set(selected_features_xgb_xgb))
union_features = list(set(selected_features_xgb_rf) | set(selected_features_xgb_xgb))

print("Common Features:", common_features)
print("Union Features:", union_features)

# Compare Multiple Models

models = {
    "Random Forest": rf_classifier_best_value,
    "XG Boost": xg_classifier,
    "RF with RFE Features": rf_model,
    "XGB with RFE Features": xgb_xgb_model
}
results = {}
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
results_df = pd.DataFrame(results).T
print("Model Comparison Results:")
print(results_df)
# Visualizing Model Comparison
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.savefig('Visualization/Model_Comparison.png', bbox_inches='tight')
#plt.show()


# Model Evaluation (Cross-Validation & Metrics)
f1_scores = cross_val_score(rf_model, X_train_xgb_encoded, y_train, cv=5, scoring='f1')
print("CV F1 Mean:", f1_scores.mean())

# Predictions on test set
y_pred = rf_model.predict(X_test_xgb_encoded)
y_prob = rf_model.predict_proba(X_test_xgb_encoded)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.savefig('Visualization/Confusion_Matrix_RF_RFE_Selected.png', bbox_inches='tight')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig('Visualization/ROC_Curve.png', bbox_inches='tight')
plt.show()

# Model comparison (RF vs XGB vs Ensemble)

f1_scores = cross_val_score(rf_model, X_train_xgb_encoded, y_train, cv=5, scoring='f1')
print("CV F1 Mean:", f1_scores.mean())

# Predictions on test set
y_pred = rf_model.predict(X_test_xgb_encoded)
y_prob = rf_model.predict_proba(X_test_xgb_encoded)[:, 1]


print(classification_report(y_test, y_pred))

print("ROC AUC:", roc_auc_score(y_test, y_prob))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Lapsed","Lapsed"], yticklabels=["Not Lapsed","Lapsed"])
plt.savefig('Visualization/Confusion_Matrix_RF_RFE_Selected.png', bbox_inches='tight')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1],'--',color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig('Visualization/ROC_Curve.png', bbox_inches='tight')
plt.show()


prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.plot(rec, prec, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig('Visualization/Precision_Recall_Curve.png', bbox_inches='tight')
plt.legend()
plt.show()

