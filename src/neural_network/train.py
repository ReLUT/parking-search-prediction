from model import *


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

    return parkingRecords, trackingRecords


def preprocess_data(parkingRecords, trackingRecords):
    # Preprocess data here
    parkingRecords = parkingRecords
    ########## CREATE DATASET
    df_model = trackingRecords[trackingRecords["distToParkingSpot"] < 1500].copy()

    df_model_1 = prepare_data(df_model)

    df_model_5 = reduce_freq(df_model, 5)
    df_model_5 = adjust_dur_to_previous(df_model_5)
    df_model_5 = prepare_data(df_model_5)

    df_model_10 = reduce_freq(df_model, 10)
    df_model_10 = adjust_dur_to_previous(df_model_10)
    df_model_10 = prepare_data(df_model_10)

    df_model_15 = reduce_freq(df_model, 15)
    df_model_15 = adjust_dur_to_previous(df_model_15)
    df_model_15 = prepare_data(df_model_15)

    df_model_20 = reduce_freq(df_model, 20)
    df_model_20 = adjust_dur_to_previous(df_model_20)
    df_model_20 = prepare_data(df_model_20)

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

    X_train_1, y_train_1, X_eval_1, y_eval_1, X_test_1, y_test_1 = train_test(
        df_model_1, variables=trained_vars
    )
    X_train_5, y_train_5, X_eval_5, y_eval_5, X_test_5, y_test_5 = train_test(
        df_model_5, variables=trained_vars
    )
    X_train_10, y_train_10, X_eval_10, y_eval_10, X_test_10, y_test_10 = train_test(
        df_model_10, variables=trained_vars
    )
    X_train_15, y_train_15, X_eval_15, y_eval_15, X_test_15, y_test_15 = train_test(
        df_model_15, variables=trained_vars
    )
    X_train_20, y_train_20, X_eval_20, y_eval_20, X_test_20, y_test_20 = train_test(
        df_model_20, variables=trained_vars
    )

    X_train = pd.concat([X_train_1, X_train_5, X_train_10, X_train_15, X_train_20])
    y_train = pd.concat([y_train_1, y_train_5, y_train_10, y_train_15, y_train_20])
    X_eval = pd.concat([X_eval_1, X_eval_5, X_eval_10, X_eval_15, X_eval_20])
    y_eval = pd.concat([y_eval_1, y_eval_5, y_eval_10, y_eval_15, y_eval_20])
    X_test = pd.concat([X_test_1, X_test_5, X_test_10, X_test_15, X_test_20])
    y_test = pd.concat([y_test_1, y_test_5, y_test_10, y_test_15, y_test_20])

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_eval.reset_index(drop=True, inplace=True)
    y_eval.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_eval, y_test, y_train, y_eval, y_test


def train_model(X_train, y_train):
    model_1s = nn_model(X_train, y_train, verbose=0, n1=128, n2=32, dropout=0.5)
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
    X_train, X_eval, X_test, y_train, y_eval, y_test = preprocess_data(trackingRecords)
    model = train_model(X_train, y_train)
    save_model(model)
    model_pred_eval(model, X_eval, y_eval, X_test, y_test)
