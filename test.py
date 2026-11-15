# Sends one invoice to the service. Start the service first (see README, "How to run").

import requests

url = 'http://localhost:9696/predict'

invoice = {
    "business_code": "U001",
    "cust_number": "200744019",
    "invoice_currency": "USD",
    "cust_payment_terms": "NAA8",
    "total_open_amount": 11173.02,
    "posting_date": "2020-03-19",
    "baseline_create_date": "2020-03-19",
    "due_in_date": "2020-04-03"
}

response = requests.post(url, json=invoice).json()
print(response)
