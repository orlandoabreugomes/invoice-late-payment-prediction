# Trains the final model (XGBoost) and saves it to model.bin.
# Same steps as notebook.ipynb, sections 1 and 5.2.

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Ignore all warnings

import warnings
warnings.filterwarnings('ignore')


# Parameters

output_file = 'model.bin'

numerical   = ["log_amount", "days_allowed", "days_to_due", "due_month", "due_weekday", "due_on_weekend"]
categorical = ["business_code", "invoice_currency", "cust_payment_terms", "cust_number"]

xgb_params = {
    'eta': 0.5,
    'max_depth': 7,
    'min_child_weight': 2,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
num_boost_round = 200


# 1.1 Download


import kagglehub

path = kagglehub.dataset_download('rajattomar132/payment-date-dataset')
data = path + '/dataset.csv'

print('Reading the dataset...')
df = pd.read_csv(data)


# 1.2 Clean

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df = df.drop_duplicates()

for c in ['document_create_date.1', 'area_business', 'document_type', 'invoice_id', 'posting_id']:
    del df[c]


# 1.3 Dates

def format_date(s):
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64")

    s = s.astype("string").str.split("_").str[0]

    is_ymd = s.str.fullmatch(r"\d{8}").fillna(False).astype(bool)

    mdy = pd.to_datetime(s.mask(is_ymd),  format="%m/%d/%Y", errors="coerce")
    ymd = pd.to_datetime(s.where(is_ymd), format="%Y%m%d",   errors="coerce")
    return mdy.fillna(ymd)


for c in ['clear_date', 'posting_date', 'document_create_date', 'due_in_date', 'baseline_create_date']:
    df[c] = format_date(df[c])


# 1.4 Target: only paid invoices have a real answer

df = df[df.isopen == 0].copy()
df["late"] = (df.clear_date > df.due_in_date).astype(int)

del df['clear_date']
del df['isopen']


# 1.5 Features

df["days_allowed"]   = (df.due_in_date - df.baseline_create_date).dt.days
df["days_to_due"]    = (df.due_in_date - df.posting_date).dt.days
df["due_month"]      = df.due_in_date.dt.month
df["due_weekday"]    = df.due_in_date.dt.dayofweek
df["due_on_weekend"] = (df.due_in_date.dt.dayofweek >= 5).astype(int)
df["log_amount"]     = np.log1p(df["total_open_amount"])


# 1.6 Split: 80% full train, 20% test

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

y_full_train = df_full_train.late.values
y_test       = df_test.late.values


# 5.2 Final model: XGBoost trained on train + validation

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(df_full_train[categorical + numerical].to_dict(orient='records'))
X_test       = dv.transform(df_test[categorical + numerical].to_dict(orient='records'))

features = list(dv.get_feature_names_out())

dfull_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest       = xgb.DMatrix(X_test,       label=y_test,       feature_names=features)

print('training the final model...')
model = xgb.train(xgb_params, dfull_train, num_boost_round=num_boost_round)

y_pred = model.predict(dtest)
print('test AUC = %.4f' % roc_auc_score(y_test, y_pred))


# Save the DictVectorizer and the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print('the model is saved to', output_file)
