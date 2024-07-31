import pandas as pd
import numpy as np

df = pd.DataFrame({'col1': [1,2,3,4,5,6,6,7]})
df['col2'] = df['col1'].shift(-1)
print(df)
x_forecast = np.array(df.drop(['col2'], axis=1))[-1:]
print(x_forecast)