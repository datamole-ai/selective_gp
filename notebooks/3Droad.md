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

## 3D Road Network (North Jutland, Denmark) Data Set
https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland,+Denmark)

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
```

```python
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt", names=["OSM_ID", "LONGITUDE", "LATITUDE", "ALTITUDE"])
data = data.astype(np.float32)
data = data.drop(columns=['OSM_ID'])
```

```python
plt.scatter(data['LONGITUDE'], data['LATITUDE'], c=data['ALTITUDE'], cmap='terrain')
plt.show()
```

```python
X_train, X_test, y_train, y_test = train_test_split(data[["LONGITUDE", "LATITUDE"]], data["ALTITUDE"], test_size=0.3, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
```

```python
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
```

```python
# subset
X_train = X_train[:10000]
y_train = y_train[:10000]

X_valid = X_valid[:2000]
y_valid = y_valid[:2000]

X_test = X_test[:10000]
y_test = y_test[:10000]
```

```python
scaler = StandardScaler()
X_train = torch.from_numpy(scaler.fit_transform(X_train)).to(torch.float64)
X_valid = torch.from_numpy(scaler.transform(X_valid)).to(torch.float64)
X_test = torch.from_numpy(scaler.transform(X_test)).to(torch.float64)

scaler = StandardScaler()
y_train = torch.from_numpy(scaler.fit_transform(np.array(y_train).reshape(-1, 1))).to(torch.float64)
y_valid = torch.from_numpy(scaler.transform(np.array(y_valid).reshape(-1, 1))).to(torch.float64)
y_test = torch.from_numpy(scaler.transform(np.array(y_test).reshape(-1, 1))).to(torch.float64)
```

```python
y_train = torch.transpose(y_train, -1, 0)[0]
y_valid = torch.transpose(y_valid, -1, 0)[0]
y_test = torch.transpose(y_test, -1, 0)[0]
```

```python
X_train.type()
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
    "pre_epochs": [2500],
    "ppp_epochs": [1500],
    "post_epochs": [1000],
    "M": [2**11, 2**12, 2**13],
    "learning_rate": [0.002],
    "prior_rate": [0.6, 0.5, 0.4, 0.3], # alpha
    "variational_pp_probs": [1.0, 0.6, 0.2] # set before ppp training (ppp = poisson point process)
}
```

```python
import itertools
settings = (dict(zip(parameters, x)) for x in itertools.product(*parameters.values()))
```

```python
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

    # print_info(gp)

    print("Pre-fitting")
    model.fit(X=X_train, Y=y_train, max_epochs=setting["pre_epochs"])
    gp.variational_point_process.probabilities = setting["variational_pp_probs"]
    # print_info(gp)

    print("Pruning")
    model.fit_score_function_estimator(
        X=X_train,
        Y=y_train,
        learning_rate=setting["learning_rate"],
        max_epochs=setting["ppp_epochs"],
        n_mcmc_samples=8,
    )

    # print_info(gp)

    remove_points(gp)
    gp.variational_point_process.probabilities = 1.0

    # print_info(gp)

    print("Post-fitting")
    model.fit(X=X_train, Y=y_train, max_epochs=setting["post_epochs"])

    # print_info(gp)

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
results.to_csv("3Droad_results2.csv", index=False)
```

```python
results
```

```python
results
```

```python

```
