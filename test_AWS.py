# Sends one invoice to the service deployed on AWS Elastic Beanstalk
# and shows the details of the communication.

import socket
from urllib.parse import urlparse

import requests

url = 'http://invoice-late-payment-env.eba-pv5wp9cb.us-east-1.elasticbeanstalk.com/predict'

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

host = urlparse(url).hostname

print('Request')
print('  URL:        ', url)
print('  Server IP:  ', socket.gethostbyname(host))
print('  Invoice:    ', invoice)

response = requests.post(url, json=invoice)

print()
print('Response')
print('  Status code:', response.status_code)
print('  Server:     ', response.headers.get('Server'))
print('  Date:       ', response.headers.get('Date'))
print('  Time:        %.0f ms' % (response.elapsed.total_seconds() * 1000))
print('  Prediction: ', response.json())
