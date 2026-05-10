# pytimetk for Time Series Analysis in Python

pytimetk combines intuitive feature engineering with interactive visualizations for enhanced time series analysis.

### `pytimetk` for Time Series Analysis in Python
#### pytimetk combines intuitive feature engineering with interactive visualizations for enhanced time series analysis.
`pytimetk` is a time series analysis and forecasting in Python inspired by the R package `timetk`.

It has some nifty tools for feature engineering and visualization (be careful, the creators spell it the British way "visuali**s**ation"). The visualization is my favorite part. It uses plotly under the hood so the graphs are interactive --- unfortunately the graphs are static in medium.


Feature Engineering is really easy in `pytimetk` --- including more advanced concepts like Fast Fourier Transforms. The time-based filtering and aggregation is ok --- but I will probably use Pandas for this in the future.

#### Let's look at a time series dataset using `pytimetk`.
I'm using random data here.



<figcaption>A pretty plot</figcaption>


#### Feature Engineering
Create lagged features, rolling averages, and Fourier terms for machine learning.


#### Resample and Aggregate
We can use `pytimetk` to aggregate the time series data by week or filter data for a specific time period.



<figcaption>Zooming in on part of the data</figcaption>


#### Forecast Evaluation
Now let's get to some actual work. We will create sample data and evaluate forecasting performance.


With the set up complete, we can move to the ML.



By comparison, here is the same plot with matplotlib. It is clear which visualization is better. There is a weird thing where `pytimetk` connects the first and last points in a straight line.


`pytimetk` is based on Pandas so we can use any machine learning models (e.g., XGBoost or LSTM) from sklearn that we want. I like the Fourier transforms for seasonal decomposition to analyze trends and periodic patterns.

`pytimetk` doesn't do ARIMA or SARIMA. I guess this is fine since we have other tools that do this well. Just odd.

Let's look at some real data. This data comes from ERCOT.


We can filter the data to just values in 2025.


You can see a big increase in demand on Jan 7 when a large storm came through Texas.



And this plot is still better than the plot from matplotlib.



### So what?
I liked `pytimetk` more than I thought I would. Maybe it is because I love R and the syntax is similar. I really like how it creates interactive plotly graphs.

Code for this project is available on [GitHub](https://github.com/kylejones200/time_series/blob/main/pytimekt.ipynb).
