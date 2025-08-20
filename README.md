### Policy Lapse Prediction – Life Insurance
##  Project Overview

Policy lapses are a critical challenge in the life insurance industry, leading to revenue loss and disrupted customer relationships. This project builds machine learning models to predict policy lapses using customer and policy-related data.

We experimented with multiple models – Random Forest, XGBoost, and their feature-selected variants – and compared them using accuracy, precision, recall, and F1 score.

##  Features

Data preprocessing (missing values,handling null , feature Engineering ,  categorical encoding, feature scaling).
Feature selection using Recursive Feature Elimination (RFE).
Model building with Random Forest & XGBoost.
Model comparison across multiple metrics.
Correlation heatmap and EDA visualizations.
Exportable trained models for deployment.

## Model Comparison Results
    Model	                Accuracy	Precision	Recall	F1 Score
    Random Forest	        0.690	        0.743	0.740	0.742
    XGBoost	                0.717	        0.769	0.756	0.762
    RF with RFE Features	0.688	        0.742	0.737	0.739
    XGB with RFE Features	0.717	        0.769	0.756	0.762

# XGBoost (with or without RFE) performed the best overall.

## Tech Stack

Python 3.12+
Pandas, NumPy, Scikit-learn – preprocessing, feature engineering, ML models
XGBoost – boosting algorithm
Seaborn, Matplotlib – visualization

##  Project Structure
Policy_Lapse_LifeInsurance/
│── policy_lapse.py             # Main script
│── requirements.txt            # Dependencies
│── Visualization/              # Saved plots (heatmap, metrics, etc.)
│── README.md                   # Documentation
│── dataset/                    # Raw & processed datasets (ignored in git)
│── model/

## Installation & Usage

    Clone the repo:

        git clone https://github.com/Decadent-tech/Policy_Lapse_LifeInsurance
        cd Policy_Lapse_LifeInsurance


    Install dependencies:

        pip install -r requirements.txt


    Run the script:

        python policy_lapse.py

## Deploy the model with Streamlit

        ![Policy Lapse Dashboard](https://github.com/Decadent-tech/Policy_Lapse_LifeInsurance/blob/main/Dashboard.jpg)

## Next Steps

Tune hyperparameters for better accuracy.
Explore SHAP/LIME for model interpretability.
Test with real-world policy datasets.

## Contribution
Contributions, issues, and feature requests are welcome!
Feel free to open a PR or raise an issue.
