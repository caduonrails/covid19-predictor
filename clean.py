import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv('data/covid_19_clean_complete.csv', parse_dates=['Date'])

# Preprocessing
## Fill missing values
data[['Province/State']] = data[['Province/State']].fillna('')
data[['Confirmed', 'Deaths', 'Recovered']] = data[
        ['Confirmed', 'Deaths', 'Recovered']].fillna(0)

## Fix datatypes
data['Recovered'] = data['Recovered'].astype(int)

print(data.head())
