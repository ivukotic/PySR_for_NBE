from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysr import PySRRegressor

df = pd.read_fwf('mass.mas03', usecols=(2, 3, 11),
                 widths=(1, 3, 5, 5, 5, 1, 3, 4, 1, 13, 11, 11,
                         9, 1, 2, 11, 9, 1, 3, 1, 12, 11, 1),
                 skiprows=41, header=None,
                 index_col=False)
df.columns = ('n', 'Z',  'avEbind')

# Extrapolated values are indicated by '#' in place of the decimal place, so
# the avEbind column won't be numeric. Coerce to float and drop these entries.
df['avEbind'] = pd.to_numeric(df['avEbind'], errors='coerce')
df = df.dropna()
df['A'] = df.n + df.Z
X = df[['A', 'n', 'Z']]
y = df[['avEbind']]

model = PySRRegressor(
    update=False,
    multithreading=True,
    niterations=50,  # default is 40
    populations=50,  # default is 15
    binary_operators=["plus", "mult", "pow"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "log",
        "odd(x) = isodd(x) ? 1.0f0 : -1.0f0"
    ],
    extra_sympy_mappings={'special': lambda x, y: x**2 + y},
    # model_selection="best", best is default "accuracy"
    # loss="f(x, y) = (x - y)^2",  # this is default,
    constraints={'pow': (-1, 3), 'mult': (3, 3), 'cos': 5, 'sin': 5}
)

model.fit(X, y)
print(model)

eqs = model.equations
be = eqs.score.idxmax()
l = eqs.loss[be]
s = eqs.score[be]
e = eqs.sympy_format[be]
ef = eqs.lambda_format[be]
print("best result:\nloss:", l, '  score:', s, '  eq:', e)

df["pred"] = ef(X)
df["res"] = df.avEbind - df.pred

pdf = df.drop(['avEbind', 'A'], axis=1)
tp = pdf.set_index(['Z', 'n']).res.unstack(0)

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
z_min = pdf.res.min()
z_max = pdf.res.max()
print(z_min, z_max)

# c = ax.pcolormesh(tp, cmap= cm.coolwarm, norm=colors.LogNorm(vmin=z_min, vmax=z_max))
c = ax.pcolormesh(tp, norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=z_min, vmax=z_max, base=10),
                  cmap=cm.coolwarm, shading='auto')

ax.set_title('optimized')
# set the limits of the plot to the limits of the data
ax.axis([pdf.Z.min(), pdf.Z.max(), pdf.n.min(), pdf.n.max()])
fig.colorbar(c, ax=ax)

ax.set_xlabel('Z')
ax.set_ylabel('N')

plt.show()
plt.savefig("prediction.png", dpi=150)
