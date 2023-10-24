# Parking Search Prediction
Parking search prediction model based on the ground truth data collected in start2park research project.

<img align="top" src="docs/images/start2park_logo.jpg" width="30%" height="30%">

## When does the search begin?

This model is designed to predict the starting point of a parking search within a GPS trajectory. 
It leverages ground truth data, which includes the precise onset of the parking search, to train the 
predictive algorithm.

![whenSeachBegins.png](docs%2Fimages%2FwhenSeachBegins.png)

## Ground Truth Data Collection

Our `start2park` mobile app simplifies the collection of ground truth data for parking search. 
User-friendly and efficient, the app's main screen is designed for quick interactions.

![app_mainScreen.jpg](docs%2Fimages%2Fapp_mainScreen.png)

**Four-Step Data Collection:**

We collect time and location data in four easy steps:

1. Origin of the journey
2. Starting point of the search
3. Parking spot
4. Final walking destination

This comprehensive approach ensures that the model is trained on high-quality data.

![sampleJourney.jpg](docs%2Fimages%2FsampleJourney.jpg)

## Model Architecture

Our model leverages deep learning neural network technology, comprising two hidden 
layers optimized for both speed and accuracy.

![modelArchetecture.png](docs%2Fimages%2FmodelArchetecture.png)

**Technical Specifications:**
1. Input Layer: Receives spatial-temporal features from GPS trajectories.
2. Hidden Layers: Consists of activation functions for non-linear transformations.
3. Output Layer: Produces the likelihood of the onset of a parking search.


## Parking Search Visualization App

Discover and analyze the parking search pattern within a single journey using our Visualization App. It complements our machine learning model by offering a seamless way to validate and visualize your GPS trajectory data.

**How It Works:**

1. Upload Your CSV File: Use the upload button in the app to select your CSV file, which should contain data for a single journey.
2. Automatic Validation: The app will validate your file against a set of predefined criteria.
3. Reveal The Parking Search Pattern: If your file is validated successfully, you'll be given two options:
    - Download: Retrieve a GeoJSON file with predicted parking search labels.
    - Dashboard: Access our Visualization Dashboard to interactively explore the parking search route.

**Requirements:**

Before uploading your file, please make sure it meets the following guidelines:

Required Columns: The CSV must include these exact columns: lon, lat, timestamp, and speed_kmh. Additional columns are allowed.

1. Data Types:
   * Longitude (`lon`) and Latitude (`lat`): Decimal format
   * Timestamp (`timestamp`): Recognized datetime format (e.g., 'YYYY-MM-DD HH:MM:SS')
   * Speed (`speed_kmh`): Non-negative and in km/h

2. File Constraints:
   * Maximum file size: 10 MB
   * Minimum and Maximum Rows: 2 to 10,000

3. Temporal Guidelines:
   * Single journey, not exceeding 30 days
   * Median time frequency between points should be less than 20 seconds


Failure to meet these criteria will result in the file not being processed. 
You'll receive an error message specifying what needs to be corrected.

## Predicting Parking Search Routes for Multiple Journeys

For those interested in large-scale analysis across multiple journeys, our model offers 
a batch processing feature. This capability is invaluable for urban planners, researchers, 
and data scientists who aim to build aggregate statistics and visualize parking search 
patterns across multiple scenarios.

**How It Works:**

1. Clone or Download the Repository: The first step is to clone or download the entire 
repository to ensure that you have all necessary files, including the model and helper scripts.
2. Set Up the Environment: Follow the guidelines to set up your Python environment 
as described in [here](examples/README.md).
3. Download and Open the Jupyter Notebook: 
Open the [example Jupyter notebook](examples/ParkingSearchPrediction.ipynb) 
designed to guide you through the process.
4. Prepare Your Dataset: Ensure your dataset is formatted correctly according to the 
guidelines specified in the "Visualization App" section. Note that your dataset can be 
extensive, supporting up to millions of individual trips.
5. Adjust the Notebook: The notebook comes with pre-defined column names and parameters. 
Make sure to adjust these according to your new dataset before running the notebook.
6. Run the Notebook: Execute the modified notebook to preprocess your data, make parking 
search predictions, and generate aggregate statistics.
7. Visualize Sample Journeys: The notebook also provides functionalities to visualize 
a selection of individual journeys, enabling you to better understand the model’s predictions.
8. Export Results: The notebook will allow you to export your findings and aggregate 
statistics for further analysis or visualization.

**Benefits:**

1. Scalability: Designed for large datasets with the capacity to analyze millions of trips.
2. Aggregate Statistics: Gain comprehensive statistical insights into parking search behavior 
across a wide array of scenarios.
3. Sample Visualization: Explore sample journeys for a nuanced understanding of individual routes.
5. Export Capability: Conveniently export your results for further analysis or to integrate 
with other data sets.



## Project Partners

Data collection in `star2park` reasearch project is a result of a joint work of following partners:

- Frankfurt University of Applied Sciences, Frankfurt am Main
- Fluxguide Ausstellungssysteme GmbH, Wien (Österreich)
- Bliq GmbH, Braunschweig

## Acknowledgements
This parking search prediction model is the result of the research project 
[start2park](https://www.start2park.com) conducted by the 
[Research Lab for Urban Transport](https://www.frankfurt-university.de/en/about-us/faculty-1-architecture-civil-engineering-geomatics/research-institute-ffin/specialist-groups-of-the-ffin/specialist-group-new-mobility/relut/). 
This research project is funded by the German Federal Ministry for Digital and Transport (BMDV) 
funding under the “mFUND” funding program. [FKZ: 19F2114A]