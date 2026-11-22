# StockStats Alpha Lab

> **Website:** [StockStats Alpha Lab](https://aitelkob.github.io/stockstats-alpha-lab/)

A Python research toolkit for working with OHLCV market data, engineering technical indicators with `stockstats`, creating time-series labels, evaluating signals, backtesting strategies with configurable costs, and reviewing risk metrics. It includes reusable modules, Jupyter notebooks, tests, and an interactive Streamlit dashboard. Results depend on the data and assumptions supplied by the user; the project does not establish predictive performance.

## 🚀 Key Features

- **Technical indicators**: Trend, momentum, volatility, volume, and oscillator features via `src/indicators.py`
- **Time-series labels and signal evaluation**: Forward returns, triple-barrier labels, and information coefficient helpers in `src/labeling.py`
- **Strategy backtesting**: Configurable commission and slippage in `src/backtest.py`
- **Model pipelines**: Time-ordered validation split and scikit-learn pipelines in `src/models.py`
- **Risk metrics and plots**: Return, drawdown, VaR/CVaR, and reporting helpers in `src/risk_metrics.py` and `src/plots.py`
- **Interactive dashboard**: Streamlit app in `streamlit_app.py`

## 📁 Project Structure

```
stockstats-alpha-lab/
├── notebooks/                    # Jupyter notebooks for the complete workflow
│   ├── 01_data_loading.ipynb
│   ├── 02_feature_engineering_stockstats.ipynb
│   ├── 03_signal_eval_crossval.ipynb
│   ├── 04_strategy_backtests.ipynb
│   └── 05_risk_reports.ipynb
├── src/                         # Core Python modules
│   ├── data.py                  # yfinance/polars/pandas loaders
│   ├── indicators.py            # stockstats wrapper & feature engineering
│   ├── labeling.py              # forward returns & triple-barrier labels
│   ├── models.py                # sklearn pipelines & XGBoost
│   ├── backtest.py              # vectorized backtesting engine
│   └── plots.py                 # matplotlib reporting functions
├── tests/                       # Comprehensive test suite
│   ├── test_indicators.py       # Indicator validation tests
│   ├── test_labeling.py         # Labeling discipline tests
│   └── test_backtest.py         # Backtesting engine tests
├── pyproject.toml              # Dependencies and configuration
├── requirements.txt            # Dashboard/runtime dependencies
├── Makefile                    # Common development commands
├── streamlit_app.py            # Interactive dashboard
└── README.md                   # This file
```

## 📚 User Guides

### For Everyone (Non-Technical Users)
- **[🚀 Quick Start Guide](QUICK_START_GUIDE.md)** - Get up and running in 5 minutes
- **[📊 Complete User Guide](DASHBOARD_USER_GUIDE.md)** - Detailed explanation of every feature
- **[🎯 Strategy Parameters Guide](STRATEGY_PARAMETERS_GUIDE.md)** - How to use strategy parameters and risk management

### For Developers
- **[🔧 Makefile Guide](MAKEFILE_GUIDE.md)** - How to use all the development commands
- **[✨ Enhancement Summary](ENHANCEMENT_SUMMARY.md)** - Overview of all advanced features

## 🛠️ Installation

### Prerequisites

- Python 3.10+ (as declared in `pyproject.toml`)
- pip or conda

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aitelkob/stockstats-alpha-lab.git
cd stockstats-alpha-lab

# Install dependencies
pip install -e .

# Run tests to verify installation
pytest tests/

# Launch the Streamlit dashboard
streamlit run streamlit_app.py

# Or launch the research notebooks
jupyter notebook notebooks/
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
isort src/ tests/
ruff check src/ tests/
```

## 🎯 Core Demo: Feature Engineering

The heart of this project is the elegant feature engineering using `stockstats`:

```python
import pandas as pd
from stockstats import StockDataFrame as SDF

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    sdf = SDF.retype(df.copy())  # expects columns: open, high, low, close, volume
    
    # Trend & momentum (one-line indicators!)
    sdf['close_10_sma']; sdf['close_20_ema']; sdf['macd']; sdf['rsi_14']
    sdf['close_10_roc']; sdf['mstd_20']; sdf['boll']; sdf['kdjk']; sdf['kdjd']; sdf['kdjj']
    
    # Volatility & range features
    sdf['atr_14']; sdf['cr']; sdf['wr_14']
    
    # Log-returns (built-in)
    sdf['log-ret']
    
    return pd.DataFrame(sdf)
```

## 📊 Workflow Overview

### 1. Data Loading (`01_data_loading.ipynb`)
- Clean yfinance integration for OHLCV data
- Multi-ticker batch loading with progress tracking
- Data quality validation and error handling
- Basic exploratory data analysis

### 2. Feature Engineering (`02_feature_engineering_stockstats.ipynb`)
- Technical indicator engineering with `stockstats`
- Validation against reference implementations
- Information Coefficient analysis
- Feature selection and ranking

### 3. Signal Evaluation (`03_signal_eval_crossval.ipynb`)
- Forward return labeling with time discipline
- Triple-barrier labeling for classification
- Time-series signal evaluation
- IC analysis and model evaluation

### 4. Strategy Backtesting (`04_strategy_backtests.ipynb`)
- **RSI + Trend filter**: Long when `rsi_14 < 30` AND `close > close_200_sma`
- **MACD crossover**: With volatility-based position sizing
- Vectorized backtesting with realistic costs
- Performance metrics and risk analysis

### 5. Risk Reports (`05_risk_reports.ipynb`)
- Professional tearsheets and performance attribution
- Drawdown analysis and risk metrics
- Monthly returns heatmaps
- Interactive visualizations

## 🎲 Example Strategies

### RSI + Trend Filter Strategy
```python
def rsi_trend_strategy(df):
    # Long when RSI oversold AND price above trend
    long_condition = (df['rsi_14'] < 30) & (df['close'] > df['close_200_sma'])
    short_condition = (df['rsi_14'] > 70) & (df['close'] < df['close_200_sma'])
    
    signals = pd.Series(0, index=df.index)
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals
```

### MACD Crossover with Volatility Sizing
```python
def macd_crossover_strategy(df):
    # Position size inversely proportional to volatility
    base_signals = macd_crossover(df)
    vol_adjustment = df['atr_14'].rolling(20).median() / df['atr_14']
    sized_signals = base_signals * vol_adjustment.clip(0.5, 2.0)
    
    return sized_signals
```

## 📈 Performance Metrics

The backtesting engine calculates comprehensive performance metrics:

- **Returns**: Total, annualized, excess returns
- **Risk**: Volatility, Sharpe ratio, Sortino ratio
- **Drawdown**: Maximum drawdown, Calmar ratio
- **Trading**: Hit rate, turnover, transaction costs
- **Risk Management**: VaR, Expected Shortfall

## 🧪 Testing & Validation

### Indicator Validation
```python
# Cross-check RSI calculation against reference
validation_results = validate_indicators_against_reference(df, reference_library="ta")
assert validation_results['rsi_14'] == True  # Correlation > 0.99
```

### Time-Series Discipline
```python
# Ensure no look-ahead bias in labeling
forward_returns = forward_return_label(df, horizon=5)
assert forward_returns.iloc[-5:].isna().all()  # Last 5 values must be NaN
```

### Comprehensive Test Suite
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_indicators.py
pytest tests/test_labeling.py
pytest tests/test_backtest.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set API keys for additional data sources
export ALPHA_VANTAGE_API_KEY="your_key_here"
export QUANDL_API_KEY="your_key_here"
```

### Custom Configuration
```python
# Modify backtesting parameters
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,      # 0.1% commission
    slippage=0.0005,       # 0.05% slippage
    max_position_size=0.1   # 10% max position
)
```

## 📚 Dependencies

### Core Dependencies
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `stockstats>=0.2.0` - Technical indicators
- `yfinance>=0.2.0` - Market data
- `scikit-learn>=1.3.0` - Machine learning
- `xgboost>=1.7.0` - Gradient boosting

### Visualization
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.15.0` - Interactive plots

### Development
- `pytest>=7.4.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `ruff>=0.0.280` - Linting

## 🚨 Important Notes

### StockStats Caveats
This project acknowledges and addresses known issues with the `stockstats` library:

- **Validation**: Cross-checking against reference implementations (e.g., `ta` library)
- **Unit Tests**: Spot checks for critical indicators like RSI and MACD
- **Documentation**: Clear disclaimers about library limitations

### Data Quality
- **OHLCV Validation**: Automatic checks for price relationships
- **Missing Data**: Robust handling of gaps and errors
- **Time Zones**: Consistent datetime handling

### Reproducibility
- **Random Seeds**: Fixed seeds for reproducible results
- **Dependency Locking**: Exact version specifications
- **Environment**: Docker support for consistent environments

## 🤝 Contributing

Bug reports, reproducible research examples, documentation improvements, and focused code changes are welcome. Open an [issue](https://github.com/aitelkob/stockstats-alpha-lab/issues) to discuss a bug or proposal, then submit a pull request with a clear description and the commands or data assumptions needed to reproduce the change. Run `pytest tests/` before opening a pull request when practical. Please note that no project license is currently included.

## 📄 License

No `LICENSE` file is currently present in this repository. Do not assume a license grant; contact the maintainer with licensing questions.

## 🙏 Acknowledgments

- **StockStats**: [jealous/stockstats](https://github.com/jealous/stockstats) for the technical indicators library
- **yfinance**: [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) for market data access
- **InsiderFinance**: [Trading Indicators Tutorial](https://wire.insiderfinance.io/how-to-get-the-7-most-popular-trading-indicators-using-stockstats-in-python-8814bc6e6923) for inspiration

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: aitelkobyoussef@gmail.com
- **LinkedIn**: [[youssef](https://www.linkedin.com/in/youssef-ait-el-kob/)]
- **GitHub**: [GitHub Profile](https://github.com/aitelkob)

---

**Disclaimer**: This project is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and consider consulting with financial professionals before making investment decisions.
