import os
import pandas as pd


### Format Tests:


def check_file_extension(filename):
    _, ext = os.path.splitext(filename)
    if ext.lower() != ".csv":
        raise ValueError("Uploaded file is not a CSV file.")


def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError("Error reading the CSV file. Please check its format.") from e

### Content Tests:


def check_header_names(data):
    expected_headers = ["lon", "lat", "timestamp", "speed_kmh"]
    for col in expected_headers:
        if col not in list(data.columns):
            raise ValueError(
                f"{col} is not included in the header."
            )


def check_data_types(data):
    for index, row in data.iterrows():
        try:
            lon = float(row["lon"])
            lat = float(row["lat"])
            timestamp = pd.to_datetime(
                row["timestamp"]
            )  # This will work for many standard datetime formats.
            speed_kmh = float(row["speed_kmh"])

            if speed_kmh < 0:
                raise ValueError(f"Negative speed detected at row {index + 1}")

        except ValueError:
            raise ValueError(f"Invalid data type at row {index + 1}")


def check_gps_bounds(data):
    if not ((-180 <= data["lon"].min()) and (data["lon"].max() <= 180)):
        raise ValueError("Longitude values should be between -180 and 180 degrees.")

    if not ((-90 <= data["lat"].min()) and (data["lat"].max() <= 90)):
        raise ValueError("Latitude values should be between -90 and 90 degrees.")


def check_speed_bounds(data, max_speed=300):
    if data["speed_kmh"].max() > max_speed:
        raise ValueError(
            f"Detected speeds above {max_speed} km/h, which is unexpected."
        )


### Constraint Tests:


def check_file_size(filename, max_size_mb=10):
    file_size = os.path.getsize(filename) / (1024 * 1024)  # Get file size in MB
    if file_size > max_size_mb:
        raise ValueError(f"File size exceeds the limit of {max_size_mb} MB.")


def check_row_count(data, min_rows=2, max_rows=10000):
    row_count = data.shape[0]
    if not (min_rows <= row_count <= max_rows):
        raise ValueError(
            f"Number of rows should be between {min_rows} and {max_rows}. Found {row_count} rows."
        )


def check_trip_duration(data, max_duration=pd.Timedelta("30 days")):
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    duration = data["timestamp"].max() - data["timestamp"].min()
    if duration > max_duration:
        raise ValueError(f"Trip duration exceeds the limit of {max_duration}.")


def check_timestamp_order(data):
    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not in ascending order.")


def check_duplicates(data):
    if data.duplicated().any():
        raise ValueError("Duplicate rows found in the data.")


def check_missing_values(data):
    if data.isnull().any().any():
        raise ValueError("Missing values found in the data.")


### Sanity Checks:


def check_static_points(data, max_duration=pd.Timedelta("1 hour")):
    # Getting the times where the speed is 0
    static_times = data[data["speed_kmh"] == 0]["timestamp"]
    if not static_times.empty:
        time_diff = static_times.diff().fillna(pd.Timedelta("0 days"))
        if (time_diff > max_duration).any():
            raise ValueError("Detected static points exceeding the allowable duration.")


def check_timestamp_gaps(data, max_gap=pd.Timedelta("1 hour")):
    gaps = data["timestamp"].diff().fillna(pd.Timedelta("0 days"))
    if (gaps > max_gap).any():
        raise ValueError("Detected irregular gaps in timestamps.")


def check_median_time_interval(data, max_interval_seconds=20):
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    time_intervals = data["timestamp"].diff().dropna().dt.total_seconds()
    median_interval = time_intervals.median()
    if median_interval > max_interval_seconds:
        raise ValueError(
            f"Median time frequency of {median_interval} seconds is too high. It should be less than {max_interval_seconds} seconds."
        )


def validate_csv(
    file_path,
    max_size_mb=10,
    min_rows=2,
    max_rows=10000,
    max_duration=pd.Timedelta("30 days"),
    static_max_duration=pd.Timedelta("1 hour"),
    timestamp_gap=pd.Timedelta("1 hour"),
    max_interval_seconds=20,
    verbose=False
):
    """
    Validate a given CSV file against a set of predefined checks.
    """
    # Format Tests
    check_file_extension(file_path)
    data = load_csv(file_path)

    # Content Tests
    check_header_names(data)
    check_data_types(data)
    check_gps_bounds(data)
    check_speed_bounds(data)

    # Constraint Tests
    check_file_size(file_path, max_size_mb)
    check_row_count(data, min_rows, max_rows)
    check_trip_duration(data, max_duration)
    check_timestamp_order(data)
    check_duplicates(data)
    check_missing_values(data)

    # Sanity Checks
    check_static_points(data, static_max_duration)
    check_timestamp_gaps(data, timestamp_gap)
    check_median_time_interval(data, max_interval_seconds)

    if verbose:
        print("CSV validation successful!")
