# Creating the prediction environment

There are two ways to create the environment for parking search prediction.

## Using YAML file
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

## Using a new ENV and installing Dependencies

Create a new environment:
```shell
conda conda create -n parkPred -c conda-forge
```

Activate this environment:
```shell
conda activate parkPred
```

Install the dependencies:
- pandas==2.0.*
- geopandas=0.12.*
- seaborn=0.12.*
- tensorflow=2.11.*
- keras=2.11.*
- contextily=1.3.*
- folium=0.14.*


# Notebook with dynamic map output
You can open the notebook using nbviewer in [this link](https://nbviewer.org/github/ReLUT/parking-search-prediction/blob/main/examples/ParkingSearchPrediction.ipynb) to see the dynamic map outputs.

