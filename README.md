# ML Project 1


- Run and make our best submissions (Ridge Regression on dataset $\mathcal{D}_\star$)
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