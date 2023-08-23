# Set up training environment

To set up the required environment:

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you haven't already.
2. Navigate to the project's root directory.
3. Create a new Conda environment with:
`conda env create -f environment.yml`
4. Activate the new environment with:
`conda activate geoTensor`

# Check on the input CSV file

Here's a list of tests to ensure the quality and integrity of the uploaded CSV file:

**Format Tests**:
1. File Extension: Ensure that the uploaded file has a .csv extension.
2. Valid CSV Format: Ensure that the file is correctly delimited (typically by commas) and has valid rows and columns.

**Content Tests**:
1. Header Names: Check if the lon, lat, timestamp, and speed_kmh are included in the headers.
2. Data Type Checks:
    lon and lat should be valid floating point numbers.
    timestamp can be a UNIX timestamp or a standardized datetime format.
    speed_kmh should be a non-negative floating point number.
3. GPS Bounds:
    Longitude (lon) should be between -180 and 180.
    Latitude (lat) should be between -90 and 90.
4. Speed Check: Ensure speed_kmh is within reasonable bounds, e.g., 0 to 300 km/h (or whatever the maximum speed you expect).

**Constraint Tests**:

1. File Size Limit: Depending on your server or application constraints, you might want to limit the file size. For instance, files larger than 10MB might be rejected.
2. Number of Rows: Decide on a minimum and maximum number of rows that make sense for a trajectory. For example:
    Minimum: 2 rows (you need at least 2 points to have a trajectory)
    Maximum: 10,000 rows (or any number that seems practical for processing).
3. Trip Duration: Using the timestamp column, compute the duration of the trip. If the trip duration is too short (e.g., less than a minute) or too long (e.g., over a month), it might be an error or not relevant for your application.
4. Order of Timestamps: Ensure that the timestamps are in ascending order. If not, it might indicate an issue with the data recording.
5. Duplicates: Check for duplicate rows, as these might distort your model's predictions.
6. Missing Values: Ensure there are no missing values in the dataset.

**Sanity Checks**:
1. Static Points: If the speed is 0 km/h for a very long time, it might be an indication that the device was off or not moving. Depending on your use case, you might want to flag or exclude these from analysis.
2. Speed vs. Distance: Check if the calculated distance between two consecutive points matches the recorded speed and timestamp. Huge discrepancies might indicate incorrect data.
3. Timestamp Frequency: Check if there are irregular gaps in timestamps, which might suggest missing data or recording errors.
4. Average Timestamp Frequency: Check if the median time frequency is less than 20 seconds.

**User Feedback**:
Error Messages: Whenever a test fails, provide a clear and helpful error message to the user so they know how to correct the issue. For instance, if the latitudes are out of bounds, inform the user which rows have incorrect latitudes and provide the accepted range.

