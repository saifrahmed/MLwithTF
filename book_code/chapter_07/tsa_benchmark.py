import os

import pandas as pd
import requests

import numpy as np

import matplotlib.pyplot as plt

# This code is inspired from Google's notebook in the link below:
# https://github.com/googledatalab/notebooks/blob/master/samples/TensorFlow/Machine%20Learning%20with%20Financial%20Data.ipynb

API_KEY = '91x_HX5rSkHJT7xNMgUb'

codes = ["WSE/OPONEO_PL", "WSE/VINDEXUS", "WSE/WAWEL", "WSE/WIELTON", "WIKI/SNPS"]
start_date = '2010-01-01'
end_date = '2015-01-01'
order = 'asc'
column_index = 4

data_specs = 'start_date={}&end_date={}&order={}&column_index={}&api_key={}'.format(start_date, end_date, order,
                                                                                    column_index, API_KEY)

base_url = "https://www.quandl.com/api/v3/datasets/{}/{" \
           "}/data.json?" + data_specs

output_path = os.path.realpath('../../datasets/TimeSeries')

if not os.path.isdir(os.path.realpath('../../datasets/TimeSeries')):
    os.makedirs(output_path)

closings = pd.DataFrame()

for code in codes:
    code_splits = code.split("/")
    stock_exchange = code_splits[0]
    index = code_splits[1]

    stock_data = requests.get(base_url.format(stock_exchange, index)).json()
    dataset_data = stock_data['dataset_data']
    data = np.array(dataset_data['data'])
    closings[index] = pd.Series(data[:, 1].astype(float))
    closings[index + "_scaled"] = closings[index] / max(closings[index])

closings.fillna(method='ffill')
closings.describe(include='all')

fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(15)

for code in codes:
    index = code.split("/")[1]
    _ = plt.plot(closings[index], label=index)

_ = plt.legend(loc='upper right')

plt.show()

fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(15)

for code in codes:
    index = code.split("/")[1]
    _ = plt.plot(closings[index + "_scaled"], label=index + "_scaled")

_ = plt.legend(loc='upper right')

plt.show()