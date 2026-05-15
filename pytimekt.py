"""Generated from Jupyter notebook: pytimekt

Magics and shell lines are commented out. Run with a normal Python interpreter."""



def main():
    # --- code cell ---

    df.head()


    # --- code cell ---

    import pandas as pd
    import pytimetk as tk

    # Load data
    df = pd.read_csv("ercot_load_data.csv")
    df["date"] = pd.to_datetime(df["date"])


    # Plot the time series using pytimetk's plot_timeseries method
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


    # --- code cell ---

    """
    Feature Engineering
    Create lagged features, rolling averages, and Fourier terms for machine learning.
    """
    import pandas as pd
    import pytimetk as tk

    # Feature Engineering
    # Add rolling mean and standard deviation for a window of 7 days
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

    # Add rolling mean and standard deviation for a window of 14 days
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

    # Add Fourier series for seasonality
    rolled_df = tk.augment_fourier(
        rolled_df_14,
        date_column="date",  # Specify the date column
    )
    rolled_df.tail()


    # --- code cell ---

    """
    Time-Based Filtering
    Filter data for a specific time period.
    """

    # Filter data for the year 2021
    df_filtered = tk.filter_by_time(df, date_column="date", start_date="2025-01-01")

    # Plot the filtered data using pytimetk's plot_timeseries method
    fig = tk.plot_timeseries(
        df_filtered,
        date_column="date",
        value_column="values",
        facet_ncol=1,
        x_axis_date_labels="%b %Y",  # Adjusted for better visualization within a year
        engine="plotly",
        title="Filtered Time Series (2025)",
    )

    fig.write_image("filtered_time_series_2025.png")
    fig.show()


    # --- code cell ---

    import pandas as pd
    import pytimetk as tk
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    df.sort_values(by="date", inplace=True)

    # Step 2: Feature Engineering
    # Add lagged features for 1, 2, and 3 days
    df_features = tk.augment_lags(
        df, date_column="date", value_column="values", lags=[1, 2, 3]
    )

    # Add rolling mean and standard deviation for 7 and 14 days
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

    # Add Fourier series for seasonality
    df_features = tk.augment_fourier(
        df_features,
        date_column="date",  # Specify the date column
    )

    # Drop rows with NaN values (resulting from lagged features or rolling stats)
    df_features = df_features.dropna()

    # Verify the resulting DataFrame
    print(df_features.head())

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
    model = RandomForestRegressor(random_state=123)
    model.fit(X_train, y_train)
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    # Step 4: Combine Actual and Predicted Values for Plotting
    test["Predicted"] = y_pred
    # Melt data for pytimetk plotting (long format)
    plot_df = test.melt(
        id_vars="date",
        value_vars=["values", "Predicted"],
        var_name="Series",
        value_name="Values",
    )
    # Plot using pytimetk
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


    # --- code cell ---

    plot_df.to_csv("plot.csv")


    # --- code cell ---

    df.tail()


    # --- code cell ---

    from sklearn.ensemble import RandomForestRegressor

    train_columns = ["values"]

    X = train[train_columns]
    y = train[["Weekly_Sales"]]

    model = RandomForestRegressor(random_state=123)
    model = model.fit(X, y)


    Code
    predicted_values = model.predict(future[train_columns])
    future["y_pred"] = predicted_values

    future.head(10)


    # --- code cell ---

    # Ensure indices of predictions and test target match
    assert len(y_pred) == len(y_test), (
        "Mismatch in lengths of predictions and actual values."
    )
    assert list(test.index) == list(test.index), (
        "Mismatch in indices of predictions and actual values."
    )


    # --- duplicate code cell omitted (identical to earlier cell) ---


    # --- code cell ---

    # Adjust plotting for clarity
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


    # --- code cell ---

    from sklearn.linear_model import LinearRegression

    # Train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    print(f"Linear Regression MSE: {mse_lr:.4f}")


    # --- code cell ---

    # Check for obvious alignment issues
    print(f"First few test dates: {test['date'].head()}")
    print(f"First few prediction dates: {test.index[:5]}")

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


    # --- code cell ---


    # --- code cell ---

    df.drop_duplicates(inplace=True)


    # --- code cell ---

    df.shape


    # --- code cell ---

    df.set_index("date", inplace=True)


    # --- code cell ---

    df.sort_index(inplace=True)


    # --- duplicate code cell omitted (identical to earlier cell) ---


    # --- code cell ---

    df.to_csv("ercot_load_data.csv")


if __name__ == "__main__":
    main()
