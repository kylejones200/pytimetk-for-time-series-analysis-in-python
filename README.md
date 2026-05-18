# PyTimeTK for Time Series Analysis in Python

This project demonstrates using PyTimeTK for time series analysis and manipulation.

## Business context

`pytimetk` is a time series analysis and forecasting in Python inspired by the R package `timetk`.

It has some nifty tools for feature engineering and visualization (be careful, the creators spell it the British way "visualisation"). The visualization is my favorite part. It uses plotly under the hood so the graphs are interactive --- unfortunately the graphs are static in medium.

Feature Engineering is really easy in `pytimetk` --- including more advanced concepts like Fast Fourier Transforms. The time-based filtering and aggregation is ok --- but I will probably use Pandas for this in the future.

## Article

Medium article: [PyTimeTK for Time Series Analysis in Python](https://medium.com/@kylejones_47003/pytimetk-for-time-series-analysis-in-python-92f725352d99)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # PyTimeTK analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- PyTimeTK features to compute
- Window size for analysis
- Output settings

## PyTimeTK Features

PyTimeTK provides:
- Time-based grouping: `summarize_by_time()`
- Padding: `pad_by_time()`
- Rolling operations: Time-aware rolling functions
- Time series manipulation: Date/time operations

## Caveats

- By default, generates synthetic time series data.
- PyTimeTK requires proper datetime indexing.
- Full functionality requires pytimetk package installation.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).