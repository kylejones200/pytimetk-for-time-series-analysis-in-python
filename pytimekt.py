"""Generated from Jupyter notebook: pytimekt

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import pandas as pd
import pytimetk as tk
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main() -> None:
    df.head()

    df = pd.read_csv("ercot_load_data.csv")

    df["date"] = pd.to_datetime(df["date"])

    fig = tk.plot_timeseries(
        df,
        date_column="date",
        value_column="values",
        facet_ncol=1,
        x_axis_date_labels="%Y",
        engine="plotly",
        title="ERCOT Load Data Time Series",
    )

    fig.write_image("time_series_plot.png")

    fig.show()

    "\nFeature Engineering\nCreate lagged features, rolling averages, and Fourier terms for machine learning.\n"

    rolled_df_7 = tk.augment_rolling_apply(
        df,
        date_column="date",
        window=7,
        window_func=[
            ("rolling_mean_7", lambda x: x["values"].mean()),
            ("rolling_std_7", lambda x: x["values"].std()),
        ],
        center=False,
        threads=1,
    )

    rolled_df_14 = tk.augment_rolling_apply(
        rolled_df_7,
        date_column="date",
        window=14,
        window_func=[
            ("rolling_mean_14", lambda x: x["values"].mean()),
            ("rolling_std_14", lambda x: x["values"].std()),
        ],
        center=False,
        threads=1,
    )

    rolled_df = tk.augment_fourier(rolled_df_14, date_column="date")

    rolled_df.tail()

    "\nTime-Based Filtering\nFilter data for a specific time period.\n"

    df_filtered = tk.filter_by_time(df, date_column="date", start_date="2025-01-01")

    fig = tk.plot_timeseries(
        df_filtered,
        date_column="date",
        value_column="values",
        facet_ncol=1,
        x_axis_date_labels="%b %Y",
        engine="plotly",
        title="Filtered Time Series (2025)",
    )

    fig.write_image("filtered_time_series_2025.png")

    fig.show()

    df.sort_values(by="date", inplace=True)

    df_features = tk.augment_lags(
        df, date_column="date", value_column="values", lags=[1, 2, 3]
    )

    df_features = tk.augment_rolling_apply(
        df_features,
        date_column="date",
        window=7,
        window_func=[
            ("rolling_mean_7", lambda x: x["values"].mean()),
            ("rolling_std_7", lambda x: x["values"].std()),
        ],
        center=False,
        threads=1,
    )

    df_features = tk.augment_rolling_apply(
        df_features,
        date_column="date",
        window=14,
        window_func=[
            ("rolling_mean_14", lambda x: x["values"].mean()),
            ("rolling_std_14", lambda x: x["values"].std()),
        ],
        center=False,
        threads=1,
    )

    df_features = tk.augment_fourier(df_features, date_column="date")

    df_features = df_features.dropna()

    print(df_features.head())

    train = df_features.iloc[:-30]

    test = df_features.iloc[-30:]

    X_train = train.drop(columns=["date", "values"])

    y_train = train["values"]

    X_test = test.drop(columns=["date", "values"])

    y_test = test["values"]

    model = RandomForestRegressor(random_state=123)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")

    test["Predicted"] = y_pred

    plot_df = test.melt(
        id_vars="date",
        value_vars=["values", "Predicted"],
        var_name="Series",
        value_name="Values",
    )

    fig = tk.plot_timeseries(
        plot_df,
        date_column="date",
        value_column="Values",
        color_column="Series",
        title="Forecast vs Actual",
        x_axis_date_labels="%b %d, %Y",
        engine="plotly",
    )

    fig.write_image("forecast_vs_actual_plot.png")

    fig.show()

    plot_df.to_csv("plot.csv")

    df.tail()

    train_columns = ["values"]

    X = train[train_columns]

    y = train[["Weekly_Sales"]]

    model = RandomForestRegressor(random_state=123)

    model = model.fit(X, y)

    Code

    predicted_values = model.predict(future[train_columns])

    future["y_pred"] = predicted_values

    future.head(10)

    assert len(y_pred) == len(y_test), (
        "Mismatch in lengths of predictions and actual values."
    )

    assert list(test.index) == list(test.index), (
        "Mismatch in indices of predictions and actual values."
    )

    fig = tk.plot_timeseries(
        plot_df,
        date_column="date",
        value_column="Values",
        color_column="Series",
        title="Forecast vs Actual",
        x_axis_date_labels="%b %d, %Y",
        engine="plotly",
    )

    fig.show()

    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)

    mse_lr = mean_squared_error(y_test, y_pred_lr)

    print(f"Linear Regression MSE: {mse_lr:.4f}")

    print(f"First few test dates: {test['date'].head()}")

    print(f"First few prediction dates: {test.index[:5]}")

    plt.figure(figsize=(10, 6))

    plt.plot(test["date"], y_test, label="Actual", color="Blue")

    plt.plot(test["date"], y_pred, label="Predicted", color="Red")

    plt.xlabel("Date")

    plt.ylabel("Values")

    plt.legend()

    plt.tight_layout()

    plt.savefig("forecast_plot.png")

    plt.show()

    df.drop_duplicates(inplace=True)

    df.shape

    df.set_index("date", inplace=True)

    df.sort_index(inplace=True)

    df.to_csv("ercot_load_data.csv")


if __name__ == "__main__":
    main()
