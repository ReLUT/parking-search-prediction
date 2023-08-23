# Creating the prediction framework

Create a new environment using the provided YAML file:
```shell
conda env create -f environment.yml
```

Activate this environment:
```shell
conda activate parkPred
```

Add this environment to notebook kernels:
```shell
python -m ipykernel install --user --name parkPred
```

Open the provided jupyter notebook. You can now run the example code in 
the newly installed `parkPred` environment. Or you can try to predict
new data.


