# Parking Search Prediction
Parking search prediction model based on the ground truth data collected in start2park research project.

<img align="top" src="docs/images/start2park_logo.jpg" width="30%" height="30%">

## When does the search begin?

This model is designed to predict the starting point of a parking search within a GPS trajectory. 
It leverages ground truth data, which includes the precise onset of the parking search, to train the 
predictive algorithm.

![whenSeachBegins.png](docs%2Fimages%2FwhenSeachBegins.png)

## Ground Truth Data Collection

`start2park` is a mobile app that collects ground truth data on parking search. 
The functionality of the app is intentionally easy. 
The main screen of the app can be seen in the image below. 

![app_mainScreen.jpg](docs%2Fimages%2Fapp_mainScreen.jpg)

In four steps, we collect the time and location of 1) origin of the journey, 2) the starting point 
of the search, 3) the parking spot, and 4) the final walking destination. A sample journey
recorded by start2park app can be seen in the image below:

![sampleJourney.jpg](docs%2Fimages%2FsampleJourney.jpg)

## Model Architecture

Deep learning neural network with two hidden layers.

![modelArchetecture.png](docs%2Fimages%2FmodelArchetecture.png)



## Project Partners

`star2park` reasearch project is a result of a joint work of following partners:

- Frankfurt University of Applied Sciences, Frankfurt am Main
- Fluxguide Ausstellungssysteme GmbH, Wien (Österreich)
- Bliq GmbH, Braunschweig

## Acknowledgements
This parking search prediction model is the result of the research project 
[start2park](https://www.start2park.com) conducted by the 
[Research Lab for Urban Transport](https://www.frankfurt-university.de/en/about-us/faculty-1-architecture-civil-engineering-geomatics/research-institute-ffin/specialist-groups-of-the-ffin/specialist-group-new-mobility/relut/). 
This research project is funded by the German Federal Ministry for Digital and Transport (BMDV) 
funding under the “mFUND” funding program. [FKZ: 19F2114A]