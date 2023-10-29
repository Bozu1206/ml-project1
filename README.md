# ML Project 1


- Run and make our best submissions (Ridge Regression on dataset $\mathcal{D}_\star$). This will generate a file named `sub.csv`.
```console
$ python3 run.py
```

- Run all models (on dataset $\mathcal{D}_\star$)
```console
$ python3 run.py --all True
```

- Run all models on a specific dataset (use a `ds` factor of `0.0` for $\mathcal{D}_1$, `0.1` for $\mathcal{D}_2$ and `0.3` for $\mathcal{D}_3$)
```console 
$ python3 run.py --all True -ds [0.0, 0.1, 0.3]
```

- Do cross-validation for our best model 
```console 
$ python3 run.py -cv True
```


# Code explanation 

The required code for all models is contained in the `implementations.py` file.

Nevertheless, for enhanced convenience, we introduced a Fitter object corresponding to each model. These objects encompass helpful methods like `fit()` and `predict()`, and their implementation can be found in the `fitter.py` file.

Additionally, due to our selection of a specific feature set, we developed a JSON Parser capable of parsing a particular JSON format. This parser aids in the selection and cleaning of all the required features. This functionality is implemented purely in Python and is located in the `preprocessing.py` and `json_parser.py` files. The preprocessing file contains also the function used to resample the original dataset.

It's worth noting that the cross-validation process has been (awkwardly) implemented in the `cross_validation.py` file. Although the implementation of polynomial feature expansion exists in the polynomial_exp.py file, it was not utilized in this particular project (refer to the report for more details).

Finally, we have created a Metrics class that facilitates the straightforward computation of accuracy and the $F_1$-score.