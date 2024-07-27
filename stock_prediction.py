import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd

df = quandl.get("WIKI/GOOGL")
df = df.columns("Adj. Close")