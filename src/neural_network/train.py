from model_NN import *


def load_data():
    parkingRecords = pd.read_csv(
        "validParkingRecords.csv",
        parse_dates=[
            "rideStarted.timestamp",
            "parkingStarted.timestamp",
            "rideEnded.timestamp",
            "parkingEnded.timestamp",
        ],
    )
    trackingRecords = pd.read_csv(
        "validTripsTrackingRecords.csv", parse_dates=["timestamp"]
    )

    trackingRecords = pd.merge(trackingRecords,
                               parkingRecords[['parkingRecordId',
                                               'parkingSearchDuration [min]']])

    return parkingRecords, trackingRecords


def preprocess_data(parkingRecords, trackingRecords, freq=None, n_splits=10, fold_variation=1):
    df_model = trackingRecords[trackingRecords["distToParkingSpot"] < 1500].copy()

    if freq is None:
        df_model = prepare_data(df_model)

    else:
        df_model = reduce_freq(df_model, freq)
        df_model = adjust_dur_to_previous(df_model)
        df_model = prepare_data(df_model)
        df_model['parkingRecordId'] = df_model['parkingRecordId'] + f'{freq}s'

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

    X_train, y_train, X_test, y_test = train_test(
        df_model, variables=trained_vars, n_splits=n_splits, fold_variation=fold_variation
    )

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, y_train, X_test, y_test


def preprocess_data_FinalModel(parkingRecords, trackingRecords):
    # Preprocess data here
    parkingRecords = parkingRecords
    ########## CREATE DATASET
    df_model = trackingRecords[trackingRecords["distToParkingSpot"] < 1500].copy()

    df_model_1 = prepare_data(df_model)

    df_model_5 = reduce_freq(df_model, 5)
    df_model_5 = adjust_dur_to_previous(df_model_5)
    df_model_5 = prepare_data(df_model_5)
    df_model_5['parkingRecordId'] = df_model_5['parkingRecordId'] + '5s'

    df_model_10 = reduce_freq(df_model, 10)
    df_model_10 = adjust_dur_to_previous(df_model_10)
    df_model_10 = prepare_data(df_model_10)
    df_model_5['parkingRecordId'] = df_model_5['parkingRecordId'] + '10s'

    df_model_15 = reduce_freq(df_model, 15)
    df_model_15 = adjust_dur_to_previous(df_model_15)
    df_model_15 = prepare_data(df_model_15)
    df_model_5['parkingRecordId'] = df_model_5['parkingRecordId'] + '15s'

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

    y_flexFreq_allData = pd.concat([df_model_1, df_model_5, df_model_10, df_model_15])
    X_flexFreq_allData = y_flexFreq_allData[trained_vars]

    return X_flexFreq_allData, y_flexFreq_allData


def train_model(X_train, y_train, verbose):
    model_1s = nn_model(X_train, y_train, verbose=verbose, n1=128, n2=32, dropout=0.5)
    return model_1s


def save_model(model, path="ParkingSearchPrediction.h5"):
    model.save(path)


def model_pred_eval(model_1s, X_eval, y_eval, X_test, y_test):
    metric_df, optimal_p = best_cutoff_p(
        model_1s,
        X_eval=X_eval,
        y_eval=y_eval,
        p_low=0.50,
        p_high=0.70,
        metric_psd="PSD_mean",
        verbose=0,
    )

    print("optimal cutoff: ", optimal_p)
    y_pred = make_clean_preds(
        model_1s, X_test=X_test, y_test=y_test, optimal_p=optimal_p
    )
    results_model = predict_PSD(y_pred)
    show_eval(results_model)


if __name__ == "__main__":
    parkingRecords, trackingRecords = load_data()
    X_flexFreq_allData, y_flexFreq_allData = preprocess_data_FinalModel(parkingRecords, trackingRecords)
    model = train_model(X_flexFreq_allData, y_flexFreq_allData, verbose=1)
    save_model(model)
