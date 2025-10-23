# Makefile for StockStats Alpha Lab

.PHONY: help install install-dev test lint format clean run-notebooks run-demo

help:  ## Show this help message
	@echo "StockStats Alpha Lab - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dependencies
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run all tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint:  ## Run linting
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

clean:  ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf reports/run_*/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-notebooks:  ## Launch Jupyter notebooks
	jupyter notebook notebooks/

run-demo:  ## Run a quick demo
	python -c "from src.data import DataLoader; from src.indicators import add_basic_indicators; loader = DataLoader(); df = loader.load_single_ticker('AAPL', period='6mo'); df = add_basic_indicators(df); print(f'Demo complete! Added {len(df.columns) - 5} indicators to AAPL data.')"

run-full-pipeline:  ## Run the complete pipeline
	python -c "import sys; sys.path.append('src'); from data import DataLoader; from indicators import add_basic_indicators; from labeling import LabelingEngine; from backtest import BacktestEngine, StrategyBuilder; loader = DataLoader(); df = loader.load_single_ticker('AAPL', period='1y'); df = add_basic_indicators(df); labeler = LabelingEngine(); forward_returns = labeler.forward_return_label(df, horizon=5); signals = StrategyBuilder.rsi_trend_strategy(df); engine = BacktestEngine(); results = engine.run_backtest(df, signals, strategy_name='RSI_Trend'); print(f'Pipeline complete! Strategy return: {results[\"total_return\"]:.2%}, Sharpe: {results[\"sharpe_ratio\"]:.2f}')"

check-deps:  ## Check if all dependencies are installed
	python -c "import pandas, numpy, stockstats, yfinance, sklearn, xgboost, matplotlib, seaborn; print('All dependencies are installed!')"

setup-env:  ## Set up development environment
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make install-dev"

# Documentation
docs:  ## Generate documentation (if sphinx is installed)
	@echo "Documentation generation not implemented yet"

# Docker commands (if Dockerfile exists)
docker-build:  ## Build Docker image
	@echo "Docker build not implemented yet"

docker-run:  ## Run Docker container
	@echo "Docker run not implemented yet"

# CI/CD
ci-test:  ## Run tests for CI
	pytest tests/ --cov=src --cov-report=xml --junitxml=test-results.xml

# Release
release:  ## Create a release (placeholder)
	@echo "Release process not implemented yet"
