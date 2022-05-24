# https://github.com/MilesCranmer/PySR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysr import PySRRegressor

df = pd.read_fwf('mass.mas03', usecols=(2, 3, 11),
                 widths=(1, 3, 5, 5, 5, 1, 3, 4, 1, 13, 11, 11,
                         9, 1, 2, 11, 9, 1, 3, 1, 12, 11, 1),
                 skiprows=39, header=None,
                 index_col=False)
df.columns = ('N', 'Z',  'avEbind')

# Extrapolated values are indicated by '#' in place of the decimal place, so
# the avEbind column won't be numeric. Coerce to float and drop these entries.
df['avEbind'] = pd.to_numeric(df['avEbind'], errors='coerce')
df = df.dropna()

print(df)

X = df[['N', 'Z']]
y = df[['avEbind']]

model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",  # Custom operator (julia syntax)
    ],
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
)

model.fit(X, y)
print(model)
