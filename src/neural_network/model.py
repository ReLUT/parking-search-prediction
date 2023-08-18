import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from scipy.signal import argrelextrema


from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from keras import backend as K

import matplotlib.dates as mdates
from matplotlib.dates import date2num
from datetime import timedelta


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


def dur_show(seconds, unit="sec"):
    """
    This functions displays the time from seconds and minutes in
    MM:SS format
    """

    if unit == "min":
        seconds *= 60
    if np.isnan(seconds):
        return np.nan
    if seconds >= 60 * 60:
        hours = seconds // (60 * 60)
        seconds %= 60 * 60
        minutes = seconds // 60
        seconds %= 60
        return "%02i:%02i:%02i" % (hours, minutes, seconds)
    else:
        minutes = seconds // 60
        seconds %= 60
        return "%02i:%02i" % (minutes, seconds)


def show_eval(
    df,
    var1="psd_pred [min]",
    var2="parkingSearchDuration [min]",
    display_MAE=True,
    display_RMSE=False,
):
    """
    displays the evaluation baed on the comparison of actual and predicted psd
    """

    MAE = (df[var1] - df[var2]).apply(np.abs).mean()
    RMSE = (df[var1] - df[var2]).apply(np.square).mean() ** 0.5
    print("mean actual psd:         ", dur_show(df[var2].mean(), unit="min"))
    print("mean predictions psd:    ", dur_show(df[var1].mean(), unit="min"))
    if display_MAE:
        print("mean absolute error:     ", dur_show(MAE, unit="min"))
    if display_RMSE:
        print("root mean squared error: ", dur_show(RMSE, unit="min"))


def remaining_Time(group):
    """
    This calculates the remaining time from each waypoint to the parking time
    This will be used on the tracking record dataset
    """
    return (group.iloc[-1]["timestamp"] - group["timestamp"]).apply(
        lambda x: x.total_seconds()
    )


def train_test(
    df,
    label="status",
    trip_ID="parkingRecordId",
    variables=["distToParkingSpot", "speed_kmh"],
    random_state=0,
    train_size=0.8,
    eval_set=True,
    test_eval_ratio=0.50,
):
    """
    Parameters
    =============
    df: a dataframe of the waypoints.
    e.g., in start2park that would be trackingRecords
    variables: a list of variables that should be included in training the model
    e.g. variables are: ['status', 'distToParkingSpot','speed_kmh','time_sin',
                         'time_cos','dayOfWeek','dayOfWeek','RegioStaR',
                         'temperature','wind_speed','precipitation_height']
    eval_set: if True, an evaluation set is also created.
    test_eval_ratio: this determines what percent of the rest data should be
    allocated to test set and the rest will be allocated to evaluation test

    Return
    ==============
    train, test, eval dataframes

    """

    # trips IDs in train set
    train_ID = (
        pd.Series(df[trip_ID].unique())
        .sample(random_state=random_state, n=int(df[trip_ID].nunique() * train_size))
        .values
    )
    # Create train sets
    train = df[df[trip_ID].isin(train_ID)]

    X_train = train[variables]
    X_train = pd.get_dummies(X_train, drop_first=True).astype(float)

    y_train = df[df[trip_ID].isin(train_ID)]

    # rest of the data
    rest = df[~df[trip_ID].isin(train_ID)]

    if eval_set:
        # dividing rest set into test and eval
        test_ID = (
            pd.Series(rest[trip_ID].unique())
            .sample(
                random_state=random_state,
                n=int(rest[trip_ID].nunique() * test_eval_ratio),
            )
            .values
        )
        test = rest[rest[trip_ID].isin(test_ID)]
        evaluation = rest[~rest[trip_ID].isin(test_ID)]

        X_eval = evaluation[variables]
        X_eval = pd.get_dummies(X_eval, drop_first=True).astype(float)
        y_eval = df[~df[trip_ID].isin(np.concatenate([train_ID, test_ID]))]

        # create test set
        X_test = test[variables]
        X_test = pd.get_dummies(X_test, drop_first=True).astype(float)
        y_test = df[df[trip_ID].isin(test_ID)]

        return X_train, y_train, X_eval, y_eval, X_test, y_test

    else:
        # create test set
        X_test = rest[variables]
        X_test = pd.get_dummies(X_test, drop_first=True).astype(float)
        y_test = df[~df[trip_ID].isin(train_ID)]

        return X_train, y_train, X_test, y_test


def nn_model(
    X_train,
    y_train,
    verbose=0,
    n_epochs=5,
    n1=64,
    n2=32,
    dropout=0.5,
    balance_classes=False,
):
    """
    A neural network deep learning model to predict the probability
    of a GPS point being normal driving or searching
    """
    y_train = y_train["status"]
    y_train = y_train.apply(lambda x: x == "searching").astype(int)

    model = Sequential()
    model.add(Dense(n1, input_shape=(X_train.shape[1],), activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(n2, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    # Compile the model
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=[f1_score, "accuracy"]
    )
    # Train the model
    if balance_classes:
        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))
        model.fit(
            X_train,
            y_train,
            epochs=n_epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=verbose,
            class_weight=class_weights_dict,
        )
    else:
        model.fit(
            X_train,
            y_train,
            epochs=n_epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=verbose,
        )

    return model


# create a function to apply to each group to celan predictions
def predictions_clean(group):
    """
    Based on the raw predicted results of a classification model
    This function apply the constraint that a normal driving point cannot
    come after a searching point. So it identifies the first predicted searching point
    and mark all the rest as searching.
    """

    trip_waypoints_pred = group.copy()

    trip_waypoints_pred["status_pred_clean"] = "driving"

    if "searching" in trip_waypoints_pred["status_pred"].unique():
        search_index = trip_waypoints_pred[
            trip_waypoints_pred["status_pred"] == "searching"
        ].index[0]
        trip_waypoints_pred.loc[search_index + 1 :, "status_pred_clean"] = "searching"

    return trip_waypoints_pred


def make_clean_preds(model, X_test, y_test, optimal_p=0.5, verbose=0):
    """
    make predictions for a test set
    """

    # Make predictions
    if type(model) == RandomForestClassifier:
        y_pred_binary = model.predict(X_test)
    else:
        y_pred_binary = model.predict(X_test, verbose=verbose)
        y_pred_binary = (y_pred_binary > optimal_p).astype(int)

    y_pred = y_test.copy()
    y_pred["status_pred"] = y_pred_binary
    y_pred["status_pred"] = y_pred["status_pred"].apply(
        lambda x: "driving" if x == 0 else "searching"
    )

    # apply the function to each group and concatenate the results
    y_pred = pd.concat(
        [predictions_clean(group) for _, group in y_pred.groupby("parkingRecordId")]
    )

    # max PSD set as 15 min
    idx_extreme = y_pred[
        (y_pred["status_pred_clean"] == "searching")
        & (y_pred["remainingTime"] > 15 * 60)
    ].index
    y_pred.loc[idx_extreme, "status_pred_clean"] = "driving"

    return y_pred


def predict_PSD(y_pred):
    """
    calculates the parking search durations based on the clean predictions
    of GPS points of a trip
    """
    # Calculate predicted search duration

    psd_predictions = (
        y_pred[y_pred["status_pred_clean"] == "searching"]
        .groupby("parkingRecordId")
        .first()["remainingTime"]
        .reset_index()
    )
    psd_predictions = psd_predictions.rename(columns={"remainingTime": "psd_pred"})

    results_model = pd.merge(
        y_pred.groupby("parkingRecordId").first().reset_index(),
        psd_predictions,
        how="left",
    )
    results_model["psd_pred"] = results_model["psd_pred"].fillna(0)
    results_model["psd_pred [min]"] = results_model["psd_pred"] / 60

    return results_model


def best_cutoff_p(model, X_eval, y_eval, p_low, p_high, metric_psd="MAE", verbose=0):
    """
    This function finds the best cutoff point for binary prediction

    Parameters
    ======================

    model: the trained neural network model
    X_eval: X evaluation set
    y_eval: Y evaluation set containing all other informaition
    p_low: the low range for testing the cut off point
    p_high: the high range for testing the cut off point
    metric_psd: MAE or RMSE for parking search durations

    returns
    ======================
    optimal cutoff point

    """

    cut_p_ls = [i / 100 for i in range(int(p_low * 100), int(p_high * 100))]
    PSD_mean_ls = []
    MAE_ls = []
    RMSE_ls = []

    for cut_p in cut_p_ls:
        y_pred = make_clean_preds(
            model=model, X_test=X_eval, y_test=y_eval, optimal_p=cut_p, verbose=verbose
        )
        results_model = predict_PSD(y_pred=y_pred)

        MAE = (
            (
                results_model["psd_pred [min]"]
                - results_model["parkingSearchDuration [min]"]
            )
            .apply(np.abs)
            .mean()
        )
        RMSE = (
            results_model["psd_pred [min]"]
            - results_model["parkingSearchDuration [min]"]
        ).apply(np.square).mean() ** 0.5
        PSD_mean = results_model["psd_pred [min]"].mean()

        MAE_ls.append(int(MAE * 60))
        RMSE_ls.append(int(RMSE * 60))
        PSD_mean_ls.append(int(PSD_mean * 60))

    metric_df = pd.DataFrame(
        {"p": cut_p_ls, "MAE": MAE_ls, "RMSE": RMSE_ls, "PSD_mean": PSD_mean_ls}
    )

    axtual_test_psd_mean = int(results_model["parkingSearchDuration [min]"].mean() * 60)

    # metric: lowest MAE (or RMSE) and then PSD mean
    metric_df["PSD_diff"] = (metric_df["PSD_mean"] - axtual_test_psd_mean).apply(np.abs)

    if metric_psd in ["MAE", "RMSE"]:
        lowest_MAE = metric_df[metric_psd].min()
        min_index = metric_df[
            metric_df[metric_psd].isin(list(range(lowest_MAE, lowest_MAE + 3)))
        ]["PSD_diff"].idxmin()
        optimal_p = metric_df.loc[min_index, "p"]
    elif metric_psd == "PSD_mean":
        min_index = metric_df["PSD_diff"].idxmin()
        optimal_p = metric_df.loc[min_index, "p"]
    else:
        raise
    metric_df = pd.concat(
        [metric_df[["p"]], metric_df.applymap(dur_show).drop(columns="p")], axis=1
    )
    #     #metric: PSD mean and MAE (or RMSE)
    #     metric_df['final_metric'] = metric_df[metric_psd]+metric_df['PSD_mean']-axtual_test_psd_mean
    #     optimal_p= metric_df[metric_df['final_metric']==metric_df['final_metric'].min()]['p'].iloc[0]

    return metric_df, optimal_p


def indices(lst, item):
    return max([i for i, x in enumerate(lst) if x == item])


# create a function to apply to each group to increase frequency
def reduce_freq_group(group, freq_limit):
    trip_waypoints = group.copy()

    durs = trip_waypoints["duration_to_previous"].values.tolist()[::-1]
    keepIdx = []
    for i, val in enumerate(durs):
        if i == 0:
            keepIdx.append(True)
            continue
        lastTrueIdx = indices(keepIdx, True)
        freq = sum(durs[lastTrueIdx + 1 : i + 1])
        if freq >= freq_limit:
            keepIdx.append(True)
        else:
            keepIdx.append(False)

    keepIdxReduceVariance = pd.DataFrame(
        {"durs": durs[::-1], "keepIdx": keepIdx[::-1]}, index=trip_waypoints.index
    )
    return trip_waypoints[keepIdxReduceVariance["keepIdx"]]


def reduce_freq(dataset, freq_limit):
    """
    Parameters
    =====================
    freq_limit: should be in seconds
    dataset: a pandas dataframe containing all the waypoints and a
    column called "duration_to_previous" that defines the time to the
    last recorded waypoint in second

    """

    # apply the function to each group and concatenate the results
    df = pd.concat(
        [
            reduce_freq_group(group, freq_limit)
            for _, group in dataset.groupby("parkingRecordId")
        ]
    )

    return df


# create a function to apply to each group to increase frequency
def adjust_dur_to_previous_group(group):
    wp_temp = group.copy()

    wp_temp_diff = wp_temp[["timestamp"]].diff()
    wp_temp["duration_to_previous"] = wp_temp_diff["timestamp"].apply(
        lambda x: x.total_seconds()
    )

    wp_temp_diff_next = wp_temp["timestamp"].diff(periods=-1)
    wp_temp_diff_next = wp_temp_diff_next.apply(lambda x: x.total_seconds())
    wp_temp["duration_to_next"] = wp_temp_diff_next * -1

    wp_temp[["duration_to_previous", "duration_to_next"]] = wp_temp[
        ["duration_to_previous", "duration_to_next"]
    ].fillna(0)

    return wp_temp


def adjust_dur_to_previous(dataset):
    # apply the function to each group and concatenate the results
    df = pd.concat(
        [
            adjust_dur_to_previous_group(group)
            for _, group in dataset.groupby("parkingRecordId")
        ]
    )

    return df


def prepare_data(df, n_lagged_Vars=5):
    df_model = df[df["status"].isin(["driving", "searching"])].copy()

    # VARIABLE: remaining Time
    df_model["remainingTime"] = (
        df_model.groupby("parkingRecordId").apply(remaining_Time).values
    )

    # VARIABLE: hour
    hour = df_model["timestamp"].dt.hour
    minute = df_model["timestamp"].dt.minute
    theta = (hour + minute / 60.0) * 2 * np.pi / 24.0  # calculate the angle in radians
    df_model["time_sin"] = np.sin(theta)
    df_model["time_cos"] = np.cos(theta)

    # VARIABLEs: day, geographical zone, weather and all other
    idx_temp = df_model.index
    df_model = pd.merge(df_model, parkingRecords, how="left", on="parkingRecordId")
    df_model.index = idx_temp

    ########## CREATE LAGGED VARIABLES: Speed
    for i in range(1, n_lagged_Vars + 1):  # creates 5 lagged variables by defualt value
        df_model[f"speed_lag{i}"] = df_model.groupby("parkingRecordId")[
            "speed_kmh"
        ].shift(i)

    # Fill missing values with last observed speed value
    df_model["speed_lag1"] = df_model["speed_lag1"].fillna(df_model["speed_kmh"])
    for i in range(2, n_lagged_Vars + 1):
        df_model[f"speed_lag{i}"] = df_model[f"speed_lag{i}"].fillna(
            df_model[f"speed_lag{i-1}"]
        )

    ########## CREATE LAGGED VARIABLES: Sampling Rate
    # Group the data by parkingRecordId and calculate the difference in time
    df_model["sampRate_lag0"] = (
        df_model.groupby("parkingRecordId")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )
    for i in range(1, n_lagged_Vars + 1):  # creates 5 lagged variables by defualt value
        df_model[f"sampRate_lag{i}"] = df_model.groupby("parkingRecordId")[
            "sampRate_lag0"
        ].shift(i)
        # Fill missing values with last observed value
        df_model[f"sampRate_lag{i}"] = df_model[f"sampRate_lag{i}"].fillna(
            df_model[f"sampRate_lag{i-1}"]
        )

    ########## CREATE Location Difference: Lat and Lon diff
    df_model["lonDiff_0"] = df_model.groupby("parkingRecordId")["lon"].diff().fillna(0)
    df_model["latDiff_0"] = df_model.groupby("parkingRecordId")["lat"].diff().fillna(0)
    for i in range(1, n_lagged_Vars + 1):  # creates 5 lagged variables by defualt value
        df_model[f"lonDiff_{i}"] = df_model.groupby("parkingRecordId")[
            "lonDiff_0"
        ].shift(i)
        df_model[f"latDiff_{i}"] = df_model.groupby("parkingRecordId")[
            "latDiff_0"
        ].shift(i)
        # Fill missing values with last observed value
        df_model[f"lonDiff_{i}"] = df_model[f"lonDiff_{i}"].fillna(
            df_model[f"lonDiff_{i-1}"]
        )
        df_model[f"latDiff_{i}"] = df_model[f"latDiff_{i}"].fillna(
            df_model[f"latDiff_{i-1}"]
        )

    return df_model






