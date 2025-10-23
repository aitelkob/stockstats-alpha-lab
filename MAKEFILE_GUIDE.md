# 🛠️ Makefile Testing and Usage Guide

## 📋 **Available Commands**

Run `make help` to see all available commands:

```bash
make help
```

## 🧪 **How to Test the Makefile**

### **1. Basic Functionality Tests**

```bash
# Test help command
make help

# Test clean command (safe to run)
make clean

# Test dependency check (may fail due to XGBoost)
make check-deps
```

### **2. Core Project Tests**

```bash
# Test individual modules (bypasses XGBoost import issues)
python -c "
import sys; sys.path.append('src')
from data import DataLoader
from indicators import add_basic_indicators
from labeling import LabelingEngine
from backtest import BacktestEngine, StrategyBuilder

print('✅ All core modules import successfully')
"

# Test data loading
python -c "
import sys; sys.path.append('src')
from data import DataLoader
loader = DataLoader()
df = loader.load_single_ticker('AAPL', period='1mo')
print(f'✅ Data loading: {df.shape[0]} records')
"

# Test indicators
python -c "
import sys; sys.path.append('src')
from data import DataLoader
from indicators import add_basic_indicators
loader = DataLoader()
df = loader.load_single_ticker('AAPL', period='1mo')
df = add_basic_indicators(df)
print(f'✅ Indicators: {len(df.columns) - 5} added')
"
```

### **3. Test Individual Components**

```bash
# Test indicators only
python -m pytest tests/test_indicators.py -v

# Test labeling only
python -m pytest tests/test_labeling.py -v

# Test backtesting (some tests may fail)
python -m pytest tests/test_backtest.py -v
```

## 🚀 **How to Use the Makefile**

### **For Development Workflow**

#### **1. Initial Setup**
```bash
# Set up virtual environment
make setup-env

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
make install
```

#### **2. Daily Development**
```bash
# Run quick demo
make run-demo

# Run full pipeline
make run-full-pipeline

# Run tests
make test

# Clean up generated files
make clean
```

#### **3. Code Quality**
```bash
# Format code (if tools are installed)
make format

# Run linting (if tools are installed)
make lint
```

### **For Portfolio/Interview Demo**

#### **1. Quick Demo**
```bash
# Show the project works
make run-demo
```

**Expected Output:**
```
Demo complete! Added 22 indicators to AAPL data.
```

#### **2. Full Pipeline Demo**
```bash
# Show complete workflow
make run-full-pipeline
```

**Expected Output:**
```
Pipeline complete! Strategy return: X.XX%, Sharpe: X.XX
```

#### **3. Interactive Exploration**
```bash
# Launch Jupyter notebooks
make run-notebooks
```

## 🔧 **Troubleshooting Common Issues**

### **Issue 1: XGBoost Import Errors**
**Problem**: `XGBoost Library (libxgboost.dylib) could not be loaded`

**Solution**:
```bash
# Install OpenMP for macOS
brew install libomp

# Or skip XGBoost-dependent commands
python -c "
import sys; sys.path.append('src')
from data import DataLoader
from indicators import add_basic_indicators
# ... other imports without models.py
"
```

### **Issue 2: Missing Linting Tools**
**Problem**: `make: ruff: No such file or directory`

**Solution**:
```bash
# Install development dependencies
pip install ruff black isort

# Or skip linting commands
make test  # This works without linting tools
```

### **Issue 3: Test Failures**
**Problem**: Some tests fail due to data/calculation issues

**Solution**:
```bash
# Run only working tests
python -m pytest tests/test_indicators.py -v
python -m pytest tests/test_labeling.py -v

# Or run specific test
python -m pytest tests/test_indicators.py::TestIndicators::test_add_basic_indicators -v
```

## 📊 **Makefile Command Reference**

| Command | Purpose | Status |
|---------|---------|--------|
| `make help` | Show all commands | ✅ Working |
| `make clean` | Clean generated files | ✅ Working |
| `make check-deps` | Check dependencies | ⚠️ XGBoost issue |
| `make test` | Run all tests | ⚠️ Some failures |
| `make run-demo` | Quick demo | ⚠️ XGBoost issue |
| `make run-full-pipeline` | Complete pipeline | ⚠️ XGBoost issue |
| `make run-notebooks` | Launch Jupyter | ✅ Working |
| `make install` | Install package | ✅ Working |
| `make install-dev` | Install dev deps | ⚠️ Missing tools |
| `make format` | Format code | ⚠️ Missing tools |
| `make lint` | Run linting | ⚠️ Missing tools |

## 🎯 **Recommended Usage Patterns**

### **For Quick Testing**
```bash
# Test core functionality
python -c "
import sys; sys.path.append('src')
from data import DataLoader
from indicators import add_basic_indicators
loader = DataLoader()
df = loader.load_single_ticker('AAPL', period='1mo')
df = add_basic_indicators(df)
print(f'Success! {len(df.columns) - 5} indicators added')
"
```

### **For Development**
```bash
# Clean start
make clean

# Run working tests
python -m pytest tests/test_indicators.py tests/test_labeling.py -v

# Interactive development
make run-notebooks
```

### **For Demo/Interview**
```bash
# Show help
make help

# Show clean project structure
make clean

# Show working tests
python -m pytest tests/test_indicators.py -v

# Show interactive notebooks
make run-notebooks
```

## 🚀 **Next Steps**

1. **Fix XGBoost**: Install OpenMP to enable ML functionality
2. **Install dev tools**: Add `ruff`, `black`, `isort` for code quality
3. **Fix test issues**: Address remaining test failures
4. **Add more commands**: Extend Makefile with additional functionality

## 💡 **Pro Tips**

- Use `make help` to see all available commands
- Use `make clean` before committing to remove generated files
- Use `python -c "..."` for quick testing without XGBoost issues
- Use `make run-notebooks` for interactive exploration
- Use `python -m pytest tests/test_indicators.py -v` for focused testing

The Makefile provides a professional development workflow for your StockStats Alpha Lab project! 🎯
