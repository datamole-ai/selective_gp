---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: selective_gp
    language: python
    name: selective_gp
---

# Probabilistic Selection of Inducing Points in Sparse Gaussian Processes 

## kin40k 

- the location of a robotic arm as a function of an 8-dimensional control input

https://github.com/trungngv/fgp/tree/master/data/kin40k 

```python
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sys
import json
 
# setting path
sys.path.append('..')

from selective_gp.utils import (
    load_data, get_model, remove_points, fit_layerwise)
from selective_gp.utils.visualization import (
    plot_density, plot_samples, plot_latent)

sns.set(
    font_scale=1.5,
    style="whitegrid",
)
fig_width = 16

pd.options.display.max_colwidth = 130
torch.manual_seed(0)
```

```python
def preprocess(data):
    data[0] = data[0].str.strip()
    data[0] = data[0].apply(lambda x: ' '.join(x.split()))
    data[0] = data[0].apply(lambda x: x.split(' '))
    df = pd.DataFrame(data[0].values.tolist())
    return np.array(df).astype(np.float32)
```

```python
data = pd.read_csv("../../Diploma-Thesis-Repository/datasets/kin40k/kin40k_train_data.asc", header=None)
X_train = preprocess(data)

data = pd.read_csv("../../Diploma-Thesis-Repository/datasets/kin40k/kin40k_test_data.asc", header=None)
X_test = preprocess(data)

y_train = pd.read_csv("../../Diploma-Thesis-Repository/datasets/kin40k/kin40k_train_labels.asc", header=None)[0]
y_train = np.array(y_train)

y_test = pd.read_csv("../../Diploma-Thesis-Repository/datasets/kin40k/kin40k_test_labels.asc", header=None)[0]
y_test = np.array(y_test)
```

```python
X_data, y_data = torch.tensor(X_train).to(torch.float64), torch.tensor(y_train).to(torch.float64)
X_test, y_test = torch.tensor(X_test).to(torch.float64), torch.tensor(y_test).to(torch.float64)
```

```python
print(X_data.shape)
print(X_test.shape)
print(y_test.shape)
```

```python
X_train = X_data[:9000]
y_train = y_data[:9000]

X_valid = X_data[9000:]
y_valid = y_data[9000:]

X_test = X_test[20000:22000] 
y_test = y_test[20000:22000]
```

```python
y_train = torch.transpose(y_train, -1, 0)
y_valid = torch.transpose(y_valid, -1, 0)
y_test = torch.transpose(y_test, -1, 0)
```

```python
X_train.type()
```

```python
y_test.shape
```

```python
# https://github.com/akuhren/selective_gp/blob/155713ede4de29f8ac6a9918c53f0ff7287c382a/selective_gp/datasets/synthetic_data.py#L32

class Dataset():
    def __init__(self, X_train, X_test, Y_train, Y_test):
        
        self.task_type = "regression"
        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        
        self.input_dims = self.X_train.shape[-1]
```

```python
dataset = Dataset(X_train, X_test, y_train, y_test)
```

```python
def print_info(gp):
    print(len(gp.inducing_inputs))
    print(gp.prior_point_process.rate)
    print(gp.variational_point_process.prior)
    print(gp.variational_point_process.probabilities)
```

```python
import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(y_true, y_pred):
    return mean_squared_error(y_true, np.array(y_pred), squared=False)


def get_smse(y_true, y_pred):
    # SMSE is the mean squared error (MSE) of true and predicted values, divided by the variance of the true values
    return mean_squared_error(y_true, np.array(y_pred), squared=True) / np.var(y_true)


def get_msll(y_true, y_pred_mean, y_pred_std):
    # https://github.com/scikit-learn/scikit-learn/issues/21665
    first_term = 0.5 * np.log(2 * np.pi * y_pred_std**2)
    second_term = ((y_true - y_pred_mean) ** 2) / (2 * y_pred_std**2)

    return np.nanmean(first_term + second_term)

def get_std_coverage(y, mean, std, n_std):
    # n_std = 2 95%
    # n_std = 3 99.7%~100%
    
    upper_bound = np.less_equal(y, mean + n_std*std) 
    lower_bound = np.greater_equal(y, mean - n_std*std) 
    within_bounds = np.sum(np.logical_and(lower_bound, upper_bound))
    
    return 100 *  within_bounds / len(y)
```

"In the above example, we arbitrarily set the prior rate,  𝛼 , to 1. This (fixed) hyper-parameter denotes the strength of the prior, meaning that a higher value encourages stronger pruning. A suitable value is situation specific (e.g. are we running the model on a laptop or a GPU-cluster)."


C.3.2 Real-world data

The baselines models were trained for 5000 epochs. For the adaptive method we used npre = 2500, nPPP = 1500, npost = 1000. <br>
The prior parameters alpha needed to be configured for each dataset as more observations will tend to diminish the influence of the prior.

```python
parameters = {
    "pre_epochs": [20, 50, 100], # [2, 5, 10]
    "ppp_epochs": [50, 100, 150], # [5, 10, 15]
    "post_epochs": [20, 50, 100], # [2, 5, 10]
    "M": [2**11, 2**12, 2**13],
    "learning_rate": [0.001],
    "prior_rate": [5, 1, 0.6, 0.5, 0.4, 0.3], # alpha
    "variational_pp_probs": [1.0, 0.6, 0.2] # set before ppp training (ppp = poisson point process)
}
```

```python
import itertools
settings = (dict(zip(parameters, x)) for x in itertools.product(*parameters.values()))
```

```python jupyter={"outputs_hidden": true} tags=[]
results = pd.DataFrame(
    columns=[
        "training_parameters",
        "rmse",
        "smse",
        "msll",
        "covered",
    ]
)

for setting in settings:
    print(setting)
    M = setting["M"]
    model = get_model(dataset, n_inducing=M, scale_X=False, scale_Y=False)
    (gp,) = model.gps

    gp.inducing_inputs = X_train[np.random.permutation(X_train.shape[0])[0:M], :]
    gp.prior_point_process.rate.fill_(setting["prior_rate"])

    print_info(gp)

    print("Pre-fitting")
    model.fit(X=X_train, Y=y_train, max_epochs=setting["pre_epochs"])
    gp.variational_point_process.probabilities = setting["variational_pp_probs"]
    print_info(gp)

    print("Pruning")
    model.fit_score_function_estimator(
        X=X_train,
        Y=y_train,
        learning_rate=setting["learning_rate"],
        max_epochs=setting["ppp_epochs"],
        n_mcmc_samples=8,
    )

    print_info(gp)

    print("After pruning and points removal")
    remove_points(gp)
    gp.variational_point_process.probabilities = 1.0

    print_info(gp)

    print("Post-fitting")
    model.fit(X=X_train, Y=y_train, max_epochs=setting["post_epochs"])

    print_info(gp)

    preds = gp.forward(X_test)

    mx, vx = preds.mean, preds.variance
    vx = vx.detach().numpy()
    mx = mx.detach().numpy()

    y_data = y_test.numpy()

    rmse = get_rmse(y_data, mx)
    smse = get_smse(y_data, mx)
    msll = get_msll(y_data, mx, np.sqrt(vx))

    covered = get_std_coverage(
        y_data.reshape(1, -1)[0], mx.reshape(1, -1)[0], np.sqrt(vx), 2
    )

    print("RMSE: %.4f" % rmse)
    print("SMSE: %.4f" % smse)
    print("MSLL: %.4f" % msll)
    print("Fraction of points that lie within the 95%% interval: %.4f %%" % covered)

    results = results.append(
        {
            "training_parameters": json.dumps(setting),
            "rmse": rmse,
            "smse": smse,
            "msll": msll,
            "covered": covered
        },
        ignore_index=True,
    )
```

```python
results.to_csv("kin40k_results.csv", index=False)
```

```python
results
```

```python

```
