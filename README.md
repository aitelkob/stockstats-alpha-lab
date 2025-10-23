# StockStats Alpha Lab

> **"I built a reproducible signal-factory that engineers 50+ technical factors with `stockstats`, ranks them by out-of-sample predictive power, and composes them into simple, explainable strategies with audited backtests and risk controls."**

A comprehensive quantitative finance research platform that demonstrates signal design, rigorous experimentation, and clean engineering using the `stockstats` library. This project showcases end-to-end thinking from data loading through feature engineering to strategy backtesting and risk management.

## 🚀 Key Features

- **One-line indicators**: `df['rsi_14']`, `df['macd']`, `df['close_10_ema']` - 50+ technical indicators
- **Time-series discipline**: Proper forward return labeling with no look-ahead bias
- **Vectorized backtesting**: Realistic transaction costs, slippage, and risk controls
- **Walk-forward validation**: Time-series aware cross-validation
- **Comprehensive reporting**: Professional tearsheets and risk analysis
- **Production-ready**: Unit tests, error handling, and reproducible results

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
├── reports/                     # Generated reports and artifacts
│   └── run_YYYYMMDD/           # Timestamped run outputs
├── pyproject.toml              # Dependencies and configuration
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd stockstats-alpha-lab

# Install dependencies
pip install -e .

# Run tests to verify installation
pytest tests/

# Launch Jupyter notebooks
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
- 50+ technical indicators with `stockstats`
- Validation against reference implementations
- Information Coefficient analysis
- Feature selection and ranking

### 3. Signal Evaluation (`03_signal_eval_crossval.ipynb`)
- Forward return labeling with time discipline
- Triple-barrier labeling for classification
- Walk-forward cross-validation
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

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **StockStats**: [jealous/stockstats](https://github.com/jealous/stockstats) for the technical indicators library
- **yfinance**: [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) for market data access
- **InsiderFinance**: [Trading Indicators Tutorial](https://wire.insiderfinance.io/how-to-get-the-7-most-popular-trading-indicators-using-stockstats-in-python-8814bc6e6923) for inspiration

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: candidate@example.com
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]

---

**Disclaimer**: This project is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and consider consulting with financial professionals before making investment decisions.
