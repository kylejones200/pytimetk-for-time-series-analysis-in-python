"""Generated from Jupyter notebook: pytimetk for Time Series Analysis in Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import numpy as np
import pandas as pd
import pytimetk as tk


def create_a_simulated_time_series_dataset() -> None:
    np.random.seed(42)

    n = 500

    time = pd.date_range(start="2020-01-01", periods=n, freq="D")

    values = 100 + np.cumsum(np.random.normal(0, 1, n))

    df = pd.DataFrame({"date": time, "value": values})

    print(df.head())

    fig = tk.plot_timeseries(
        df,
        date_column="date",
        value_column="value",
        facet_ncol=1,
        x_axis_date_labels="%Y",
        engine="plotly",
        title="Simulated Time Series",
    )

    fig.show()

    fig.write_image("time_series_plot.png")


def create_a_simulated_time_series_dataset_2() -> None:
    "\nStep 2: Feature Engineering\nCreate lagged features, rolling averages, and Fourier terms for machine learning.\n"

    np.random.seed(42)

    n = 500

    time = pd.date_range(start="2020-01-01", periods=n, freq="D")

    values = 100 + np.cumsum(np.random.normal(0, 1, n))

    df = pd.DataFrame({"date": time, "value": values})

    rolled_df_7 = tk.augment_rolling_apply(
        df,
        date_column="date",
        window=7,
        window_func=[
            ("rolling_mean_7", lambda x: x["value"].mean()),
            ("rolling_std_7", lambda x: x["value"].std()),
        ],
        center=False,
        threads=1,
    )

    rolled_df_14 = tk.augment_rolling_apply(
        rolled_df_7,
        date_column="date",
        window=14,
        window_func=[
            ("rolling_mean_14", lambda x: x["value"].mean()),
            ("rolling_std_14", lambda x: x["value"].std()),
        ],
        center=False,
        threads=1,
    )

    rolled_df = tk.augment_fourier(rolled_df_14, date_column="date")

    rolled_df.tail()


def filter_data_for_the_year_2021() -> None:
    "\nStep 4: Time-Based Filtering\nFilter data for a specific time period.\n"

    df_filtered = tk.filter_by_time(
        df, date_column="date", start_date="2021-01-01", end_date="2021-12-31"
    )

    fig = tk.plot_timeseries(
        df_filtered,
        date_column="date",
        value_column="value",
        facet_ncol=1,
        x_axis_date_labels="%b %Y",
        engine="plotly",
        title="Filtered Time Series (2021)",
    )

    fig.write_image("filtered_time_series_2021.png")

    fig.show()


def step_1_create_a_simulated_dataset() -> None:
    np.random.seed(42)

    n = 500

    time = pd.date_range(start="2020-01-01", periods=n, freq="D")

    values = 100 + np.cumsum(np.random.normal(0, 1, n))

    df = pd.DataFrame({"date": time, "value": values})

    df_features = tk.augment_lags(
        df, date_column="date", value_column="value", lags=[1, 2, 3]
    )

    df_features = tk.augment_rolling_apply(
        df_features,
        date_column="date",
        window=7,
        window_func=[
            ("rolling_mean_7", lambda x: x["value"].mean()),
            ("rolling_std_7", lambda x: x["value"].std()),
        ],
        center=False,
        threads=1,
    )

    df_features = tk.augment_rolling_apply(
        df_features,
        date_column="date",
        window=14,
        window_func=[
            ("rolling_mean_14", lambda x: x["value"].mean()),
            ("rolling_std_14", lambda x: x["value"].std()),
        ],
        center=False,
        threads=1,
    )

    df_features = tk.augment_fourier(df_features, date_column="date")

    df_features = df_features.dropna()

    print(df_features.head())


def step_3_forecast_evaluation() -> None:
    train = df_features.iloc[:-100]

    test = df_features.iloc[-100:]

    X_train = train.drop(columns=["date", "value"])

    y_train = train["value"]

    X_test = test.drop(columns=["date", "value"])

    y_test = test["value"]

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")

    test["Predicted"] = y_pred

    plot_df = test.melt(
        id_vars="date",
        value_vars=["value", "Predicted"],
        var_name="Series",
        value_name="Value",
    )

    fig = tk.plot_timeseries(
        plot_df,
        date_column="date",
        value_column="Value",
        color_column="Series",
        title="Forecast vs Actual",
        x_axis_date_labels="%b %d, %Y",
        engine="plotly",
    )

    fig.write_image("forecast_vs_actual_plot.png")

    fig.show()


def main() -> None:
    create_a_simulated_time_series_dataset()
    create_a_simulated_time_series_dataset_2()
    filter_data_for_the_year_2021()
    step_1_create_a_simulated_dataset()
    step_3_forecast_evaluation()


if __name__ == "__main__":
    main()
