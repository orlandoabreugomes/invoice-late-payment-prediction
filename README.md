# invoice-late-payment-prediction

Machine Learning Zoomcamp project: predict whether an invoice will be paid after its due date.

> **Status (work in progress):** data cleaning, target, features and the logistic-regression experiments are done. Tree models, `train.py`, the API and Docker are still to do.

## Problem

Late payments hurt cash flow. Given the information available when an invoice is issued (customer, amount, currency, payment terms, dates), the model predicts the probability that the invoice will be paid **after its due date** (`late = 1`). A collections team could use it to prioritize which customers to contact first.

## Dataset

- Kaggle: [Payment Date Dataset](https://www.kaggle.com/datasets/rajattomar132/payment-date-dataset) (`rajattomar132/payment-date-dataset`).
- 50,000 invoices posted between Dec 2018 and May 2020, 19 columns. Loaded in the notebook with `kagglehub`.
- The dataset has no official data dictionary; column meanings were inferred from their names and values.

## What has been done

All steps are in [`notebook.ipynb`](notebook.ipynb).

1. **Cleaning**
   - Column names to lowercase; text values to lowercase with `_` instead of spaces.
   - **1,161 duplicated rows** (identical in every column) were removed.
   - Useless columns were dropped: `document_create_date.1`, `area_business` (empty), `document_type` (almost constant), `invoice_id` (identifier with nulls), `posting_id` (constant).
2. **Dates:** `yyyymmdd` integers and month/day/year text were converted to `datetime64`. `clear_date` (payment date) is empty for open invoices.
3. **Target `late`**
   - Only **paid invoices** (`isopen == 0`) are used: `late = 1` if `clear_date > due_in_date`, else `0`.
   - The 9,681 **open invoices** were left out of training. All of them were posted after Feb 27, 2020, and all paid invoices were posted before that date, even though payments are recorded up to May 22, 2020. This is a cut in time (the payment date is not available for the newest invoices), so "open" does not mean "late". They are kept apart as the invoices a deployed model would score.
   - `clear_date` and `isopen` were deleted afterwards to avoid leakage.
   - Result: **39,158 labeled invoices, late rate = 0.419**.
4. **Features**
   - `days_allowed` (due date minus baseline date), `days_to_due` (due date minus posting date).
   - `due_month`, `due_weekday`, `due_on_weekend`.
   - `log_amount` = `log1p(total_open_amount)`; the raw amount is very skewed (skewness 2.76 vs -0.97 after the log).
   - Categorical: `business_code`, `invoice_currency`, `cust_payment_terms`, `cust_number`.
5. **Split:** random 60% train / 20% validation / 20% test (`random_state=1`). A time-based split was also tested (see below).
6. **Model:** logistic regression with standardized numerical features (scaler fitted on the training set only) and one-hot encoded categorical features (`DictVectorizer`). Metric: ROC AUC on validation.

## Experiments

Validation AUC. The test set is used once, at the end.

| # | Experiment | Numerical features | Model | AUC, random split | AUC, time split | Notes |
|---|---|---|---|---|---|---|
| 0 | First baseline | all 6 numerical features | Logistic regression (default settings) | 0.846 | n/a | `max_iter=100` |
| 1 | Base (no calendar features) | `log_amount`, `days_allowed`, `days_to_due` | Logistic regression (`liblinear`, `max_iter=1000`) | 0.8131 | 0.8081 | |
| 2 | + `due_on_weekend` | base + `due_on_weekend` | same | 0.8443 | 0.8382 | Biggest gain |
| 3 | + `due_weekday` | base + `due_weekday` | same | 0.8258 | 0.8189 | |
| 4 | + `due_month` | base + `due_month` | same | 0.8131 | 0.8079 | No gain |
| 5 | All calendar features | base + `due_month`, `due_weekday`, `due_on_weekend` | same | 0.8463 | 0.8403 | |
| 6 | Best feature set | _fill in_ | Random forest | _fill in_ | | |
| 7 | Best feature set | _fill in_ | XGBoost | _fill in_ | | |
| 8 | Final model | _fill in_ | _fill in_ | Test AUC: _fill in_ | | Reported once |

Findings so far: a due date on a weekend is the most useful calendar feature (payment tends to slip to Monday), and `due_month` adds nothing.

**Random vs time-based split.** The time-based split trains on the oldest 60% of invoices (Dec 2018 to Aug 2019), validates on the next 20% and tests on the newest 20% (Nov 2019 to Feb 2020). Its AUC is only about 0.006 lower than the random split, and the late rate is stable over time (0.43 / 0.43 / 0.39). The random split is therefore a reasonable estimate here.

## Still to do

- [ ] Random forest and XGBoost; choose the final model.
- [ ] `train.py` (trains and saves the model, `DictVectorizer` and scaler with `pickle`).
- [ ] `predict.py` (FastAPI) and a request example.
- [ ] `Dockerfile` and `requirements.txt` (pinned versions).
- [ ] "How to run" section with screenshots of the API working.

## Library versions

scikit-learn 1.7.2, pandas 2.3.3, numpy 2.3.5, xgboost 3.4.1. (With pandas 3.x the text-cleaning step does not detect the new string dtype, so keep these versions pinned.)

## Limitations

- Training uses only invoices that were already paid. Invoices still open (posted after Feb 27, 2020) have no known outcome.
- Customers repeat across the training, validation and test sets (about 98% of validation invoices belong to a customer seen in training), so the model mostly learns each customer's habits and may do worse with new customers.
- The dataset has no official documentation; some column meanings are assumptions.

## How to run

_To be completed after `train.py`, `predict.py` and the Dockerfile are ready._
