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

## airline

- flight arrival and departure times for every commercial flight in the USA from January 2008 to April 2008

https://github.com/sods/ods/blob/main/notebooks/pods/datasets/airline-delay.ipynb 

```python
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import pods
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
# torch.cuda.set_device(1)
# torch.cuda.current_device()
# torch.cuda.empty_cache()
```

```python
data = pods.datasets.airline_delay() # set num_train num_test to download more samples than default
columns = data['covariates'] + data['response']
```

```python
X_data, y_data = torch.tensor(data['X']), torch.tensor(np.transpose(data['Y'])[0]) 
X_test, y_test = torch.tensor(data['Xtest']), torch.tensor(np.transpose(data['Ytest'])[0]) 
```

```python
print(X_data.shape)
print(X_test.shape)
```

```python tags=[]
X_data.device
```

```python
# subset
X_train = X_data[:10000]
y_train = y_data[:10000]

X_valid = X_data[10000:12000]
y_valid = y_data[10000:12000]

X_test = X_test[:2000]
y_test = y_test[:2000]

# X_train = X_data[:650000]
# y_train = y_data[:650000]

# X_valid = X_data[650000:]
# y_valid = y_data[650000:]
```

```python
scaler = StandardScaler()
X_train = torch.from_numpy(scaler.fit_transform(X_train.cpu().numpy()))#.cuda()
X_valid = torch.from_numpy(scaler.transform(X_valid.cpu().numpy()))#.cuda()
X_test = torch.from_numpy(scaler.transform(X_test.cpu().numpy()))#.cuda()
```

```python
scaler = StandardScaler()
y_train = torch.from_numpy(scaler.fit_transform(y_train.cpu().numpy().reshape(-1, 1)))#.cuda()
y_valid = torch.from_numpy(scaler.transform(y_valid.cpu().numpy().reshape(-1, 1)))#.cuda()
y_test = torch.from_numpy(scaler.transform(y_test.cpu().numpy().reshape(-1, 1)))#.cuda()
```

```python
y_train = torch.transpose(y_train, -1, 0)
y_valid = torch.transpose(y_valid, -1, 0)
y_test = torch.transpose(y_test, -1, 0)
```

```python
y_test.type()
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
    print("Number of inducing points: ", len(gp.inducing_inputs))
    print("gp.prior_point_process.rate: ", gp.prior_point_process.rate)
    # print(gp.variational_point_process.prior)
    print("gp.variational_point_process.probabilities: ", gp.variational_point_process.probabilities)
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
    "pre_epochs": [20, 50, 100],
    "ppp_epochs": [50, 100, 150], 
    "post_epochs": [20, 50, 100],
    "M": [2**11, 2**12, 2**13],
    "learning_rate": [0.3, 0.1, 0.001, 0.0001], # default 0.3
    "prior_rate": [5, 1, 0.6, 0.5, 0.4, 0.3], # alpha
    "variational_pp_probs": [1.0, 0.6, 0.2], # set before ppp training (ppp = poisson point process)
    "var_learning_rate": [0.001, 0.02, 0.2, 0.4], # default 0.02 https://github.com/akuhren/selective_gp/blob/master/selective_gp/models/base_model.py#L186
    "hp_learning_rate" : [0.001, 0.02, 0.2, 0.4], # default 0.02
    "n_mcmc_samples" : [1, 2, 3, 4] # default 1
}
```

```python
import itertools
settings = (dict(zip(parameters, x)) for x in itertools.product(*parameters.values()))
```

```python
model = get_model(dataset, n_inducing=2**11, scale_X=False, scale_Y=False)
(gp,) = model.gps

gp.inducing_inputs = X_train[np.random.permutation(X_train.shape[0])[0:2**11], :]
gp.prior_point_process.rate.fill_(0.2)

print_info(gp)
```

```python
model.device
```

```python
import time
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
    start_time = time.time() # REMOVE
    model.fit(
        X=X_train,
        Y=y_train,
        max_epochs=setting["pre_epochs"],
        var_learning_rate=setting["var_learning_rate"],
        hp_learning_rate=setting["hp_learning_rate"],
        n_mcmc_samples=setting["n_mcmc_samples"],
    )
    end_time = time.time() # REMOVE
    gp.variational_point_process.probabilities = setting["variational_pp_probs"]
    print_info(gp)
    print("time: ", end_time - start_time)  # REMOVE

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
    model.fit(
        X=X_train,
        Y=y_train,
        max_epochs=setting["post_epochs"],
        var_learning_rate=setting["var_learning_rate"],
        hp_learning_rate=setting["hp_learning_rate"],
        n_mcmc_samples=setting["n_mcmc_samples"],
    )

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
            "covered": covered,
        },
        ignore_index=True,
    )
```

```python
results.to_csv("airline_results.csv", index=False)
```

```python
results
```

```python
# CPU time:  2686.531753540039
```

```python

```
