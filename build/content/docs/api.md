---
menu: main
title: API
---

<a id="pysr.sr"></a>

# pysr.sr

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L1)

<a id="pysr.sr.install"></a>

#### install

```python
def install(julia_project=None, quiet=False)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L22)

Install PyCall.jl and all required dependencies for SymbolicRegression.jl.

Also updates the local Julia registry.

<a id="pysr.sr.PySRRegressor"></a>

## PySRRegressor Objects

```python
class PySRRegressor(BaseEstimator,  RegressorMixin)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L350)

<a id="pysr.sr.PySRRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model_selection="best", *, weights=None, binary_operators=None, unary_operators=None, procs=cpu_count(), loss="L2DistLoss()", populations=15, niterations=40, ncyclesperiteration=550, timeout_in_seconds=None, alpha=0.1, annealing=False, fraction_replaced=0.000364, fraction_replaced_hof=0.035, population_size=33, parsimony=0.0032, migration=True, hof_migration=True, should_optimize_constants=True, topn=12, weight_add_node=0.79, weight_delete_node=1.7, weight_do_nothing=0.21, weight_insert_node=5.1, weight_mutate_constant=0.048, weight_mutate_operator=0.47, weight_randomize=0.00023, weight_simplify=0.0020, crossover_probability=0.066, perturbation_factor=0.076, extra_sympy_mappings=None, extra_torch_mappings=None, extra_jax_mappings=None, equation_file=None, verbosity=1e9, update_verbosity=None, progress=None, maxsize=20, fast_cycle=False, maxdepth=None, variable_names=None, batching=False, batch_size=50, select_k_features=None, warmup_maxsize_by=0.0, constraints=None, nested_constraints=None, use_frequency=True, use_frequency_in_tournament=True, tempdir=None, delete_tempfiles=True, julia_project=None, update=True, temp_equation_file=False, output_jax_format=False, output_torch_format=False, optimizer_algorithm="BFGS", optimizer_nrestarts=2, optimize_probability=0.14, optimizer_iterations=8, tournament_selection_n=10, tournament_selection_p=0.86, denoise=False, Xresampled=None, precision=32, multithreading=None, cluster_manager=None, skip_mutation_failures=True, max_evals=None, early_stop_condition=None, **kwargs, ,)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L351)

Initialize settings for an equation search in PySR.

Note: most default parameters have been tuned over several example
equations, but you should adjust `niterations`,
`binary_operators`, `unary_operators` to your requirements.
You can view more detailed explanations of the options on the
[options page](https://astroautomata.com/PySR/#/options) of the documentation.

**Arguments**:

- `model_selection` (`str`): How to select a model. Can be 'accuracy' or 'best'. The default, 'best', will optimize a combination of complexity and accuracy.
- `binary_operators` (`list`): List of strings giving the binary operators in Julia's Base. Default is ["+", "-", "*", "/",].
- `unary_operators` (`list`): Same but for operators taking a single scalar. Default is [].
- `niterations` (`int`): Number of iterations of the algorithm to run. The best equations are printed, and migrate between populations, at the end of each.
- `populations` (`int`): Number of populations running.
- `loss` (`str`): String of Julia code specifying the loss function.  Can either be a loss from LossFunctions.jl, or your own loss written as a function. Examples of custom written losses include: `myloss(x, y) = abs(x-y)` for non-weighted, or `myloss(x, y, w) = w*abs(x-y)` for weighted.  Among the included losses, these are as follows. Regression: `LPDistLoss{P}()`, `L1DistLoss()`, `L2DistLoss()` (mean square), `LogitDistLoss()`, `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`, `PeriodicLoss(c)`, `QuantileLoss(τ)`.  Classification: `ZeroOneLoss()`, `PerceptronLoss()`, `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`, `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`, `SigmoidLoss()`, `DWDMarginLoss(q)`.
- `denoise` (`bool`): Whether to use a Gaussian Process to denoise the data before inputting to PySR. Can help PySR fit noisy data.
- `select_k_features` (`None/int`): whether to run feature selection in Python using random forests, before passing to the symbolic regression code. None means no feature selection; an int means select that many features.
- `procs` (`int`): Number of processes (=number of populations running).
- `multithreading` (`bool`): Use multithreading instead of distributed backend. Default is yes. Using procs=0 will turn off both.
- `cluster_manager` (`str`): For distributed computing, this sets the job queue
system. Set to one of "slurm", "pbs", "lsf", "sge", "qrsh", "scyld", or "htc".
If set to one of these, PySR will run in distributed mode, and use `procs` to figure
out how many processes to launch.
- `batching` (`bool`): whether to compare population members on small batches during evolution. Still uses full dataset for comparing against hall of fame.
- `batch_size` (`int`): the amount of data to use if doing batching.
- `maxsize` (`int`): Max size of an equation.
- `ncyclesperiteration` (`int`): Number of total mutations to run, per 10 samples of the population, per iteration.
- `timeout_in_seconds` (`float/int`): Make the search return early once this many seconds have passed.
- `alpha` (`float`): Initial temperature.
- `annealing` (`bool`): Whether to use annealing. You should (and it is default).
- `fraction_replaced` (`float`): How much of population to replace with migrating equations from other populations.
- `fraction_replaced_hof` (`float`): How much of population to replace with migrating equations from hall of fame.
- `population_size` (`int`): Number of individuals in each population
- `parsimony` (`float`): Multiplicative factor for how much to punish complexity.
- `migration` (`bool`): Whether to migrate.
- `hof_migration` (`bool`): Whether to have the hall of fame migrate.
- `should_optimize_constants` (`bool`): Whether to numerically optimize constants (Nelder-Mead/Newton) at the end of each iteration.
- `topn` (`int`): How many top individuals migrate from each population.
- `perturbation_factor` (`float`): Constants are perturbed by a max factor of (perturbation_factor*T + 1). Either multiplied by this or divided by this.
- `weight_add_node` (`float`): Relative likelihood for mutation to add a node
- `weight_insert_node` (`float`): Relative likelihood for mutation to insert a node
- `weight_delete_node` (`float`): Relative likelihood for mutation to delete a node
- `weight_do_nothing` (`float`): Relative likelihood for mutation to leave the individual
- `weight_mutate_constant` (`float`): Relative likelihood for mutation to change the constant slightly in a random direction.
- `weight_mutate_operator` (`float`): Relative likelihood for mutation to swap an operator.
- `weight_randomize` (`float`): Relative likelihood for mutation to completely delete and then randomly generate the equation
- `weight_simplify` (`float`): Relative likelihood for mutation to simplify constant parts by evaluation
- `crossover_probability` (`float`): Absolute probability of crossover-type genetic operation, instead of a mutation.
- `equation_file` (`str`): Where to save the files (.csv separated by |)
- `verbosity` (`int`): What verbosity level to use. 0 means minimal print statements.
- `update_verbosity` (`int`): What verbosity level to use for package updates. Will take value of `verbosity` if not given.
- `progress` (`bool`): Whether to use a progress bar instead of printing to stdout.
- `maxdepth` (`int`): Max depth of an equation. You can use both maxsize and maxdepth.  maxdepth is by default set to = maxsize, which means that it is redundant.
- `fast_cycle` (`bool`): (experimental) - batch over population subsamples. This is a slightly different algorithm than regularized evolution, but does cycles 15% faster. May be algorithmically less efficient.
- `variable_names` (`list`): a list of names for the variables, other than "x0", "x1", etc.
- `warmup_maxsize_by` (`float`): whether to slowly increase max size from a small number up to the maxsize (if greater than 0).  If greater than 0, says the fraction of training time at which the current maxsize will reach the user-passed maxsize.
- `constraints` (`dict`): dictionary of int (unary) or 2-tuples (binary), this enforces maxsize constraints on the individual arguments of operators. E.g., `'pow': (-1, 1)` says that power laws can have any complexity left argument, but only 1 complexity exponent. Use this to force more interpretable solutions.
- `nested_constraints` (`dict`): Specifies how many times a combination of operators can be nested. For example,
`{"sin": {"cos": 0}}, "cos": {"cos": 2}}` specifies that `cos` may never appear within a `sin`,
but `sin` can be nested with itself an unlimited number of times. The second term specifies that `cos`
can be nested up to 2 times within a `cos`, so that `cos(cos(cos(x)))` is allowed (as well as any combination
of `+` or `-` within it), but `cos(cos(cos(cos(x))))` is not allowed. When an operator is not specified,
it is assumed that it can be nested an unlimited number of times. This requires that there is no operator
which is used both in the unary operators and the binary operators (e.g., `-` could be both subtract, and negation).
For binary operators, you only need to provide a single number: both arguments are treated the same way,
and the max of each argument is constrained.
- `use_frequency` (`bool`): whether to measure the frequency of complexities, and use that instead of parsimony to explore equation space. Will naturally find equations of all complexities.
- `use_frequency_in_tournament` (`bool`): whether to use the frequency mentioned above in the tournament, rather than just the simulated annealing.
- `tempdir` (`str/None`): directory for the temporary files
- `delete_tempfiles` (`bool`): whether to delete the temporary files after finishing
- `julia_project` (`str/None`): a Julia environment location containing a Project.toml (and potentially the source code for SymbolicRegression.jl).  Default gives the Python package directory, where a Project.toml file should be present from the install.
- `update` (`bool`): Whether to automatically update Julia packages.
- `temp_equation_file` (`bool`): Whether to put the hall of fame file in the temp directory. Deletion is then controlled with the delete_tempfiles argument.
- `output_jax_format` (`bool`): Whether to create a 'jax_format' column in the output, containing jax-callable functions and the default parameters in a jax array.
- `output_torch_format` (`bool`): Whether to create a 'torch_format' column in the output, containing a torch module with trainable parameters.
- `tournament_selection_n` (`int`): Number of expressions to consider in each tournament.
- `tournament_selection_p` (`float`): Probability of selecting the best expression in each tournament. The probability will decay as p*(1-p)^n for other expressions, sorted by loss.
- `precision` (`int`): What precision to use for the data. By default this is 32 (float32), but you can select 64 or 16 as well.
- `skip_mutation_failures` (`bool`): Whether to skip mutation and crossover failures, rather than simply re-sampling the current member.
- `max_evals` (`int`): Limits the total number of evaluations of expressions to this number.
- `early_stop_condition` (`float`): Stop the search early if this loss is reached.
- `kwargs` (`dict`): Supports deprecated keyword arguments. Other arguments will result
in an error

**Returns**:

Initialized model. Call `.fit(X, y)` to fit your data!

<a id="pysr.sr.PySRRegressor.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L793)

Prints all current equations fitted by the model.

The string `>>>>` denotes which equation is selected by the
`model_selection`.

<a id="pysr.sr.PySRRegressor.set_params"></a>

#### set\_params

```python
def set_params(**params)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L844)

Set parameters for equation search.

<a id="pysr.sr.PySRRegressor.get_params"></a>

#### get\_params

```python
def get_params(deep=True)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L856)

Get parameters for equation search.

<a id="pysr.sr.PySRRegressor.get_best"></a>

#### get\_best

```python
def get_best(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L864)

Get best equation using `model_selection`.

**Arguments**:

- `index` (`int`): Optional. If you wish to select a particular equation
from `self.equations`, give the row number here. This overrides
the `model_selection` parameter.

**Returns**:

Dictionary representing the best expression found.

<a id="pysr.sr.PySRRegressor.fit"></a>

#### fit

```python
def fit(X, y, weights=None, variable_names=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L896)

Search for equations to fit the dataset and store them in `self.equations`.

**Arguments**:

- `X` (`np.ndarray/pandas.DataFrame`): 2D array. Rows are examples, columns are features. If pandas DataFrame, the columns are used for variable names (so make sure they don't contain spaces).
- `y` (`np.ndarray`): 1D array (rows are examples) or 2D array (rows are examples, columns are outputs). Putting in a 2D array will trigger a search for equations for each feature of y.
- `weights` (`np.ndarray`): Optional. Same shape as y. Each element is how to weight the mean-square-error loss for that particular element of y.
- `variable_names` (`list`): a list of names for the variables, other than "x0", "x1", etc.
You can also pass a pandas DataFrame for X.

<a id="pysr.sr.PySRRegressor.predict"></a>

#### predict

```python
def predict(X, index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L926)

Predict y from input X using the equation chosen by `model_selection`.

You may see what equation is used by printing this object. X should have the same
columns as the training data.

**Arguments**:

- `X` (`np.ndarray/pandas.DataFrame`): 2D array. Rows are examples, columns are features. If pandas DataFrame, the columns are used for variable names (so make sure they don't contain spaces).
- `index` (`int`): Optional. If you want to compute the output of
an expression using a particular row of
`self.equations`, you may specify the index here.

**Returns**:

1D array (rows are examples) or 2D array (rows are examples, columns are outputs).

<a id="pysr.sr.PySRRegressor.sympy"></a>

#### sympy

```python
def sympy(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L956)

Return sympy representation of the equation(s) chosen by `model_selection`.

**Arguments**:

- `index` (`int`): Optional. If you wish to select a particular equation
from `self.equations`, give the index number here. This overrides
the `model_selection` parameter.

**Returns**:

SymPy representation of the best expression.

<a id="pysr.sr.PySRRegressor.latex"></a>

#### latex

```python
def latex(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L971)

Return latex representation of the equation(s) chosen by `model_selection`.

**Arguments**:

- `index` (`int`): Optional. If you wish to select a particular equation
from `self.equations`, give the index number here. This overrides
the `model_selection` parameter.

**Returns**:

LaTeX expression as a string

<a id="pysr.sr.PySRRegressor.jax"></a>

#### jax

```python
def jax(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L987)

Return jax representation of the equation(s) chosen by `model_selection`.

Each equation (multiple given if there are multiple outputs) is a dictionary
containing {"callable": func, "parameters": params}. To call `func`, pass
func(X, params). This function is differentiable using `jax.grad`.

**Arguments**:

- `index` (`int`): Optional. If you wish to select a particular equation
from `self.equations`, give the index number here. This overrides
the `model_selection` parameter.

**Returns**:

Dictionary of callable jax function in "callable" key,
and jax array of parameters as "parameters" key.

<a id="pysr.sr.PySRRegressor.pytorch"></a>

#### pytorch

```python
def pytorch(index=None)
```

[[view_source]](https://github.com/MilesCranmer/PySR/blob/42cfea5b5179095f067619fa6566dc0ecac7c12e/pysr/sr.py#L1015)

Return pytorch representation of the equation(s) chosen by `model_selection`.

Each equation (multiple given if there are multiple outputs) is a PyTorch module
containing the parameters as trainable attributes. You can use the module like
any other PyTorch module: `module(X)`, where `X` is a tensor with the same
column ordering as trained with.

**Arguments**:

- `index` (`int`): Optional. If you wish to select a particular equation
from `self.equations`, give the row number here. This overrides
the `model_selection` parameter.

**Returns**:

PyTorch module representing the expression.

