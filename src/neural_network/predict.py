from model import *
from train import *

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

### Load New Data
def f1_score(y_true, y_pred):
    """
    F1-Score written using Keras to pass into the neural network metric
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1_val

def load_model(path="ParkingSearchPrediction.h5"):
    return tf.keras.models.load_model(path, custom_objects={"f1_score": f1_score})

### Preprocess New Data
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    # Radius of earth in kilometers. Use 6371 for kilometers or 3956 for miles. Convert to meters.
    r = 6371000
    return c * r

def prepare_data(df,
                 n_lagged_Vars=5,
                 col_ID=None,
                 col_time='timestamp',
                 col_speed='speed_kmh',
                 col_lat='lat',
                 col_lon='lon'):

    """
    Create Lag Variables for Speed and Sampling Time
    """
    # Data validation
    for col in [col_lat, col_lon, col_time, col_speed]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe.")

    # make a copy of df
    df_model = df.copy()

    # Convert timestamp column to datetime format
    df_model[col_time] = pd.to_datetime(df_model[col_time])

    # A single trajectory in the dataset
    if col_ID is None:


        # Sort the data for consistency
        df_model = df_model.sort_values(by=col_time)

        # Calculate remaining distance to parking spot
        last_lat = df_model[col_lat].iloc[-1]
        last_lon = df_model[col_lon].iloc[-1]

        df_model["distToParkingSpot"] = haversine_distance(df_model[col_lat],
                                                           df_model[col_lon],
                                                           last_lat,
                                                           last_lon)

        # Create lagged variables for speed
        for i in range(1, n_lagged_Vars + 1):
            df_model[f"speed_lag{i}"] = df_model[col_speed].shift(i)

        # Create lagged variables for sampling rate
        df_model["sampRate_lag0"] = df_model[col_time].diff().dt.total_seconds().fillna(0)
        for i in range(1, n_lagged_Vars + 1):
            df_model[f"sampRate_lag{i}"] = df_model["sampRate_lag0"].shift(i)


    # Multiple trajectories in the dataset
    else:

        # Sort the data for consistency
        df_model = df_model.sort_values(by=[col_ID, col_time])

        # Remaining time calculation
        df_model["remainingTime"] = df_model.groupby(col_ID)[col_time].transform(lambda x: (x.iloc[-1] - x).dt.total_seconds())

        # Calculate remaining distance to parking spot
        last_lat = df_model.groupby(col_ID)[col_lat].transform('last')
        last_lon = df_model.groupby(col_ID)[col_lon].transform('last')

        df_model["distToParkingSpot"] = haversine_distance(df_model[col_lat],
                                                           df_model[col_lon],
                                                           last_lat,
                                                           last_lon)

        # Create lagged variables for speed
        for i in range(1, n_lagged_Vars + 1):
            df_model[f"speed_lag{i}"] = df_model.groupby(col_ID)[col_speed].shift(i)

        # Create lagged variables for sampling rate
        df_model["sampRate_lag0"] = df_model.groupby(col_ID)[col_time].diff().dt.total_seconds().fillna(0)
        for i in range(1, n_lagged_Vars + 1):
            df_model[f"sampRate_lag{i}"] = df_model.groupby(col_ID)["sampRate_lag0"].shift(i)


    df_model = df_model.fillna(method='bfill')

    return df_model

### Predictions
def remaining_Time(df, col_ID=None, col_time='timestamp'):
    """
    This calculates the remaining time from each waypoint to the parking time
    This will be used on the tracking record dataset
    """

    if col_ID is None:
        df['remainingTime'] = (df[col_time].max()-df[col_time]).dt.total_seconds()

    else:
        # VARIABLE: remaining Time
        df['remainingTime'] = df \
            .groupby(col_ID) \
            .apply(lambda group: (group[col_time].max() - group[col_time]) \
                   .dt \
                   .total_seconds()) \
            .values

    return df


def predictions_clean(group):
    """
    Based on the raw predicted results of a classification model
    This function apply the constraint that a normal driving point cannot
    come after a searching point. So it identifies the first predicted searching point
    and mark all the rest as searching.
    """

    trip_waypoints_pred = group.copy()

    trip_waypoints_pred["y_hat_labels_clean"] = "driving"

    if "searching" in trip_waypoints_pred["y_hat_labels"].unique():
        search_index = trip_waypoints_pred[
            trip_waypoints_pred["y_hat_labels"] == "searching"
            ].index[0]
        trip_waypoints_pred.loc[search_index + 1 :, "y_hat_labels_clean"] = "searching"

    return trip_waypoints_pred


trained_vars = [
    "distToParkingSpot",
    "speed_kmh",
    "speed_lag1",
    "speed_lag2",
    "speed_lag3",
    "speed_lag4",
    "speed_lag5",
    "sampRate_lag0",
    "sampRate_lag1",
    "sampRate_lag2",
    "sampRate_lag3",
    "sampRate_lag4",
    "sampRate_lag5",
]



def make_predictions(model, X_, col_ID=None, optimal_p=0.62, verbose=0, max_search_duration=15):


    y_hat = model.predict(X_[trained_vars], verbose=verbose)
    y_hat_binary = (y_hat > optimal_p).astype(int)

    y_hat_df = X_.copy()
    y_hat_df["y_hat_p"] = y_hat
    y_hat_df["y_hat_binary"] = y_hat_binary
    y_hat_df["y_hat_labels"] = y_hat_df["y_hat_binary"].apply(
        lambda x: "driving" if x == 0 else "searching"
    )


    if col_ID is None:
        # apply the function to the journey
        y_hat_df["y_hat_labels_clean"] = "driving"

        if "searching" in y_hat_df["y_hat_labels"].unique():
            search_index = y_hat_df[y_hat_df["y_hat_labels"] == "searching"].index[0]
            y_hat_df.loc[search_index + 1 :, "y_hat_labels_clean"] = "searching"

    else:
        # apply the function to each group and concatenate the results
        y_hat_df = pd.concat(
            [predictions_clean(group) for _, group in y_hat_df.groupby(col_ID)]
        )


    # max PSD set as provided limit: default 15 min
    idx_extreme= y_hat_df[(y_hat_df['y_hat_labels_clean']=='searching') & \
                          (y_hat_df['remainingTime']>max_search_duration*60)].index
    y_hat_df.loc[idx_extreme, 'y_hat_labels_clean']='driving'



    return y_hat_df

### Make Predictions For New Data
def park_search_predict(df,
                        model_path='ParkingSearchPrediction.h5',
                        p_search=0.62,
                        col_ID=None,
                        col_time='timestamp',
                        col_speed='speed_kmh',
                        col_lat='lat',
                        col_lon='lon',
                        max_search_duration=15,
                        verbose=0):
    """
    This predicts the probability of each GPS point in a GPS trajectory being
    a parking search point. Based on the evaluation dataset, the optimal probability
    to label the GPS points as "Searching" is 0.62, which is the defualt value.
    This can also be modified if necessary.
    This model uses only speed, sampling rate, and distance to destination to
    make the predictions. The resulting dataframe contains two new columns:
        "y_hat_p": This gives the probability of the point being parking search
        "y_hat_labels": This is a binary label in ['Driving', 'Searching']

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe containing the trajectories
    model_path: str
        path to the prediction model
    p_search : float
        The cut-off probability for binary classification of GPS points
        into "Driving" and "Searching"
        default: 0.62
    col_ID : str
        Name of the column that contains the Trajectories' uniqe ID. If
        there is only a single trajectory in the dataframe, set it as None
        default: None
    col_time: str
        Name of the column that contains the timestamp of the GPS points
        default: 'timestamp'
    col_speed: str
        Name of the column that contains the speed of the GPS points in km/h
        default: 'speed_kmh'
    col_lat: str
        Name of the column that contains the latitude of the GPS points
        default: 'lat'
    col_lon: str
        Name of the column that contains the longitude of the GPS points
        default: 'lon'
    max_search_duration: int
        Maximum predicted search duration in minutes
        default: 15
    verbose: 'auto', 0, 1, or 2
        Visualizing the prediction progress.
        0 = silent, 1 = progress bar, 2 = one line per epoch.
        default: 0

    Returns
    --------
    pandas.DataFrame
        Input dataframe with two added columns:
            1) y_hat_p: predicted probabilities
            2) y_hat_labels: binary classification
    """

    # check probability value
    if 0>p_search or p_search>1:
        raise ValueError('p_search must be between 0 and 1')
    # preprocess data
    df_pred = prepare_data(df,
                           n_lagged_Vars=5,
                           col_ID=col_ID,
                           col_time=col_time,
                           col_speed=col_speed,
                           col_lat=col_lat,
                           col_lon=col_lon)

    # calculate remaining time to parking for each Point
    df_pred = remaining_Time(df_pred,
                             col_ID=col_ID,
                             col_time=col_time)
    # load prediction model
    model = load_model(path=model_path)
    # make predictions
    df_pred  = make_predictions(model,
                                X_=df_pred,
                                col_ID=col_ID,
                                optimal_p=p_search,
                                verbose=verbose,
                                max_search_duration=max_search_duration)

    return df_pred