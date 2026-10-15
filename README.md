# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is going to leave the service (churn). It covers the full workflow: data exploration, preprocessing without data leakage, training and comparing three models, selecting the best one, and serving it through a web app.

**Live app:** https://customer-churn-ml-keycheck.streamlit.app/

## Data

- **Dataset:** IBM Telco Customer Churn (7,043 customers, one row per customer).
- **Source:** https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
- **Target:** `Churn Value` (1 = left, 0 = stayed). About 26.5% of customers churned, so the data is imbalanced.
- The data file is not included in this repository. See [`data/README.md`](data/README.md) for download instructions.

Columns dropped before training:

- No predictive value: `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`.
- Data leakage (not known for a customer who has not left yet): `Churn Label`, `Churn Reason`, `Churn Score`, `CLTV`.

## Method

1. Exploratory analysis of the data (see the notebook).
2. `Total Charges` was read as text because of 11 blank values (customers with 0 months of tenure). They were filled with 0 and the column converted to numeric.
3. Train/test split (80/20, stratified on the target) **before** any preprocessing.
4. A scikit-learn `Pipeline` combines `StandardScaler` (numeric features) and `OneHotEncoder` (categorical features) with the model, so everything is learned from the training set only.
5. Class imbalance is handled with `class_weight="balanced"`.

## Results

Test set: 1,409 customers, 374 of whom churned.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression (balanced) | 0.743 | 0.510 | 0.781 | 0.617 | 0.849 |
| Decision Tree (balanced) | 0.740 | 0.506 | 0.791 | 0.617 | 0.841 |
| Random Forest (tuned) | 0.773 | 0.554 | 0.746 | 0.636 | 0.854 |

Since the three test results are close, the models were also compared with 5-fold cross-validation on the training set (mean ± std):

| Model | AUC-ROC | F1 | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.859 ± 0.012 | 0.640 ± 0.021 | 0.810 ± 0.034 | 0.529 ± 0.020 |
| Decision Tree | 0.845 ± 0.008 | 0.629 ± 0.007 | 0.813 ± 0.022 | 0.514 ± 0.014 |
| Random Forest (tuned) | 0.860 ± 0.008 | 0.648 ± 0.021 | 0.751 ± 0.030 | 0.571 ± 0.021 |

**Why not accuracy alone:** a model that predicts "no churn" for every customer already scores about 0.735, so accuracy hides how many churners are missed. Recall measures how many of the customers who actually leave are caught, Precision measures how many alerts are correct, F1 balances the two, and AUC-ROC measures how well the model ranks customers by risk regardless of the decision threshold.

**Selected model: Logistic Regression.** The three models perform equally well in cross-validation (AUC-ROC about 0.85-0.86, F1 about 0.63-0.65), so the choice rests on other criteria: it reaches a high Recall (about 0.81, on par with the Decision Tree), which matters most when missing a churner is costly, and it is the simplest to interpret and deploy. Random Forest has higher Precision (fewer false alarms) but lower Recall, a difference in operating point rather than in model quality.

Its coefficients confirm the findings of the exploratory analysis: longer tenure and longer contracts reduce churn risk, while month-to-month contracts and fiber optic internet increase it.

## Project Structure

```
customer-churn-ML/
├── data/
│   └── README.md          # dataset name + download link
├── notebooks/
│   └── MLsteps.ipynb      # EDA, preprocessing, models, evaluation
├── src/
│   ├── train.py           # trains and saves the model
│   └── predict.py         # predicts for a new customer
├── models/
│   └── churn_model.joblib # saved pipeline (preprocessing + model)
├── app/
│   └── streamlit_app.py   # the web app
├── requirements.txt
└── README.md
```

## How to Run

Developed with Python 3.13.

1. Clone the repository:

```
git clone https://github.com/moohamedkishk10/customer-churn-ML.git
cd customer-churn-ML
```

2. Download the dataset (see `data/README.md`) and place `Telco_customer_churn.xlsx` in the `data/` folder.

3. Install the libraries, train the model, and run the app:

```
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
```

`python src/train.py` prints the test metrics and saves the model to `models/churn_model.joblib`. The saved model is also included in the repository, so the app can run without retraining.

To predict for a single customer from code, use `predict_customer` in `src/predict.py`.

## Limitations

- About half of the churn alerts are false alarms (Precision about 0.5). This is the cost of favoring Recall, and the decision threshold (0.5 by default) can be raised if false alarms are expensive.
- The web app estimates `Total Charges` as tenure × monthly charges, which is only an approximation.
- The dataset is a sample of about 7,000 customers of a fictional company, so results may not carry over to a real telecom operator.
- Coefficients describe patterns in the data, not causes of churn.
- Possible improvements: tune Logistic Regression like the Random Forest, try gradient boosting, and choose the threshold from the real cost of a missed churner versus a false alarm.