#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:07:35 2024

@author: jiyeonpark
"""

import pandas as pd

#%% DATA LOAD

dr = r'/Users/jiyeonpark/Library/Mobile Documents/com~apple~CloudDocs/The Project/241208 Hongkong conference/Data/cleaned'
data = pd.read_csv(dr+'/energy.csv')

data.columns = ['year','city','district','type','jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

#일단 총계 데이터만 사용함
data = data[data['type']=='총계']

monthly_columns = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
aggregated_data = data.groupby(['year','city'])[monthly_columns].sum().reset_index()
aggregated_data.to_csv(dr+'/total_aggregated_city_energy.csv', index=False)

#%% DATA PREP
import numpy as np
from sklearn.preprocessing import MinMaxScaler
seoul_data = aggregated_data[aggregated_data['city'] == '서울특별시'].drop(columns=['city']).set_index('year')
busan_data = aggregated_data[aggregated_data['city'] == '부산광역시'].drop(columns=['city']).set_index('year')

seoul_consumption = seoul_data.values.flatten()
busan_consumption = busan_data.values.flatten()

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
seoul_consumption_scaled = scaler.fit_transform(seoul_consumption.reshape(-1, 1)).flatten()
busan_consumption_scaled = scaler.transform(busan_consumption.reshape(-1, 1)).flatten()

#%% SLIDING WINDOW
# Function to create sliding window samples
def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])  # Take the previous 'window_size' months
        y.append(data[i])  # The target is the next month's consumption
    return np.array(X), np.array(y)

WINDOW_SIZE = 12
TOTAL_MONTHS = 240

# Create sliding window samples for Seoul and Busan
X_seoul, y_seoul = create_sliding_window(seoul_consumption_scaled, WINDOW_SIZE)
X_busan, y_busan = create_sliding_window(busan_consumption_scaled, WINDOW_SIZE)














