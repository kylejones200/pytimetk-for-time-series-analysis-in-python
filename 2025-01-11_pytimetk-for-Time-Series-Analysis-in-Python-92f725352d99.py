# Description: Short example for pytimetk for Time Series Analysis in Python.



from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging
import numpy as np
import pandas as pd
import pytimetk as tk

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Create a simulated time series dataset
np.random.seed(42)
n = 500
time = pd.date_range(start="2020-01-01", periods=n, freq="D")
values = 100 + np.cumsum(np.random.normal(0, 1, n))
df = pd.DataFrame({"date": time, "value": values})

# Print the first few rows
logger.info(df.head())

# Plot the time series using pytimetk's plot_timeseries method
fig = (
    tk.plot_timeseries(
        df,
        date_column='date',
        value_column='value',
        facet_ncol=1,
        x_axis_date_labels="%Y",
        engine='plotly',
        title="Simulated Time Series"
    )
)

fig.write_image("time_series_plot.png")
fig.show()

"""
Feature Engineering
Create lagged features, rolling averages, and Fourier terms for machine learning.
"""


# Feature Engineering
# Add rolling mean and standard deviation for a window of 7 days
rolled_df_7 = tk.augment_rolling_apply(
    df,
    date_column="date",
    window=7,
    window_func=[
        ("rolling_mean_7", lambda x: x["value"].mean()),
        ("rolling_std_7", lambda x: x["value"].std()),
    ],
    center=False,
    threads=1
)

# Add rolling mean and standard deviation for a window of 14 days
rolled_df_14 = tk.augment_rolling_apply(
    rolled_df_7,
    date_column="date",
    window=14,
    window_func=[
        ("rolling_mean_14", lambda x: x["value"].mean()),
        ("rolling_std_14", lambda x: x["value"].std()),
    ],
    center=False,
    threads=1
)

# Add Fourier series for seasonality
rolled_df = tk.augment_fourier(
    rolled_df_14,
    date_column="date",  # Specify the date column
)
rolled_df.tail()

"""
Time-Based Filtering
Filter data for a specific time period.
"""

# Filter data for the year 2021
df_filtered = tk.filter_by_time(
    df,
    date_column="date",
    start_date="2021-01-01",
    end_date="2021-12-31"
)

# Plot the filtered data using pytimetk's plot_timeseries method
fig = (
    tk.plot_timeseries(
        df_filtered,
        date_column='date',
        value_column='value',
        facet_ncol=1,
        x_axis_date_labels="%b %Y",  # Adjusted for better visualization within a year
        engine='plotly',
        title="Filtered Time Series (2021)"
    )
)

fig.write_image("filtered_time_series_2021.png")
fig.show()


# Feature Engineering
# Add lagged features for 1, 2, and 3 days
df_features = tk.augment_lags(
    df,
    date_column="date",
    value_column="value",
    lags=[1, 2, 3]
)

# Add rolling mean and standard deviation for 7 and 14 days
df_features = tk.augment_rolling_apply(
    df_features,
    date_column="date",
    window=7,
    window_func=[
        ("rolling_mean_7", lambda x: x["value"].mean()),
        ("rolling_std_7", lambda x: x["value"].std())
    ],
    center=False,
    threads=1
)

df_features = tk.augment_rolling_apply(
    df_features,
    date_column="date",
    window=14,
    window_func=[
        ("rolling_mean_14", lambda x: x["value"].mean()),
        ("rolling_std_14", lambda x: x["value"].std())
    ],
    center=False,
    threads=1
)

# Add Fourier series for seasonality
df_features = tk.augment_fourier(
    df_features,
    date_column="date",  # Specify the date column
)

# Drop rows with NaN values (resulting from lagged features or rolling stats)
df_features = df_features.dropna()

# Verify the resulting DataFrame
logger.info(df_features.head())

# Forecast Evaluation
# Split into training and testing sets
train = df_features.iloc[:-100]
test = df_features.iloc[-100:]
# Prepare features and target
X_train = train.drop(columns=["date", "value"])
y_train = train["value"]
X_test = test.drop(columns=["date", "value"])
y_test = test["value"]
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
logger.info(f"Mean Squared Error: {mse:.4f}")
# Step 4: Combine Actual and Predicted Values for Plotting
test['Predicted'] = y_pred
# Melt data for pytimetk plotting (long format)
plot_df = test.melt(id_vars="date", value_vars=["value", "Predicted"], 
                    var_name="Series", value_name="Value")
# Plot using pytimetk
fig = tk.plot_timeseries(
    plot_df,
    date_column="date",
    value_column="Value",
    color_column="Series",
    title="Forecast vs Actual",
    x_axis_date_labels="%b %d, %Y",
    engine="plotly"
)
fig.write_image("forecast_vs_actual_plot.png")
fig.show()



df.sort_values(by="date", inplace=True)

# Feature Engineering
# Add lagged features for 1, 2, and 3 days
df_features = tk.augment_lags(
    df,
    date_column="date",
    value_column="values",
    lags=[1, 2, 3]
)

# Add rolling mean and standard deviation for 7 and 14 days
df_features = tk.augment_rolling_apply(
    df_features,
    date_column="date",
    window=7,
    window_func=[
        ("rolling_mean_7", lambda x: x["values"].mean()),
        ("rolling_std_7", lambda x: x["values"].std())
    ],
    center=False,
    threads=1
)

df_features = tk.augment_rolling_apply(
    df_features,
    date_column="date",
    window=14,
    window_func=[
        ("rolling_mean_14", lambda x: x["values"].mean()),
        ("rolling_std_14", lambda x: x["values"].std())
    ],
    center=False,
    threads=1
)

# Add Fourier series for seasonality
df_features = tk.augment_fourier(
    df_features,
    date_column="date",  # Specify the date column
)

# Drop rows with NaN values (resulting from lagged features or rolling stats)
df_features = df_features.dropna()

# Verify the resulting DataFrame
logger.info(df_features.head())

# Step 3: Forecast Evaluation
# Split into training and testing sets
train = df_features.iloc[:-30]
test = df_features.iloc[-30:]
# Prepare features and target
X_train = train.drop(columns=["date", "values"])
y_train = train["values"]
X_test = test.drop(columns=["date", "values"])
y_test = test["values"]
# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
logger.info(f"Mean Squared Error: {mse:.4f}")
# Step 4: Combine Actual and Predicted Values for Plotting
test['Predicted'] = y_pred
# Melt data for pytimetk plotting (long format)
plot_df = test.melt(id_vars="date", value_vars=["values", "Predicted"], 
                    var_name="Series", value_name="Values")
# Plot using pytimetk
fig = tk.plot_timeseries(
    plot_df,
    date_column="date",
    value_column="Values",
    color_column="Series",
    title="Forecast vs Actual",
    x_axis_date_labels="%b %d, %Y",
    engine="plotly"
)
fig.write_image("forecast_vs_actual_plot.png")
fig.show()

# Plot actual vs predicted in a basic plot to inspect
plt.figure(figsize=(10, 6))
plt.plot(test["date"], y_test, label="Actual", color="Blue")
plt.plot(test["date"], y_pred, label="Predicted", color="Red")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_plot.png")
plt.show()
