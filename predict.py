# Web service: receives one invoice and returns the probability of late payment.
# Run:  python predict.py            (development)
#       gunicorn --bind 0.0.0.0:9696 predict:app   (production, used in Docker)

import pickle
from datetime import date

import numpy as np
import xgboost as xgb

from flask import Flask, request, jsonify


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

features = list(dv.get_feature_names_out())


def clean(text):
    # same cleaning as in training: lowercase, "_" instead of spaces
    return str(text).strip().lower().replace(' ', '_')


def prepare_features(invoice):
    # same features as notebook.ipynb, section 1.5
    # dates arrive as text in the format YYYY-MM-DD
    due      = date.fromisoformat(invoice['due_in_date'])
    baseline = date.fromisoformat(invoice['baseline_create_date'])
    posting  = date.fromisoformat(invoice['posting_date'])

    return {
        'business_code':      clean(invoice['business_code']),
        'cust_number':        clean(invoice['cust_number']),
        'invoice_currency':   clean(invoice['invoice_currency']),
        'cust_payment_terms': clean(invoice['cust_payment_terms']),
        'log_amount':         float(np.log1p(float(invoice['total_open_amount']))),
        'days_allowed':       (due - baseline).days,
        'days_to_due':        (due - posting).days,
        'due_month':          due.month,
        'due_weekday':        due.weekday(),
        'due_on_weekend':     int(due.weekday() >= 5),
    }


app = Flask('invoice-late-payment-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    invoice = request.get_json()

    try:
        X = dv.transform([prepare_features(invoice)])
    except (KeyError, ValueError, TypeError) as e:
        # a field is missing or a date/amount is in the wrong format
        return jsonify({'error': 'invalid invoice: %s' % e}), 400

    dmatrix = xgb.DMatrix(X, feature_names=features)
    late_probability = float(model.predict(dmatrix)[0])

    result = {
        'late_probability': round(late_probability, 4),
        'late': bool(late_probability >= 0.5),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
