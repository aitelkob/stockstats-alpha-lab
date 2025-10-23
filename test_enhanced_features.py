"""
Comprehensive Test Runner for Enhanced Features

This script runs all tests for the enhanced StockStats Alpha Lab features
including advanced indicators, risk metrics, optimization, and benchmarking.
"""

import sys
import subprocess
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def run_test_suite():
    """Run the complete test suite for enhanced features."""
    
    print("🚀 StockStats Alpha Lab - Enhanced Features Test Suite")
    print("=" * 60)
    
    # Test modules to run
    test_modules = [
        "tests.test_advanced_indicators",
        "tests.test_risk_metrics", 
        "tests.test_optimization",
        "tests.test_benchmarking",
        "tests.test_streamlit_app"
    ]
    
    # Core modules (existing)
    core_modules = [
        "tests.test_indicators",
        "tests.test_labeling", 
        "tests.test_backtest"
    ]
    
    all_modules = core_modules + test_modules
    
    results = {}
    
    print(f"\n📊 Running tests for {len(all_modules)} modules...")
    print("-" * 40)
    
    for module in all_modules:
        print(f"\n🧪 Testing {module}...")
        
        try:
            # Run pytest for the specific module
            result = subprocess.run(
                [sys.executable, "-m", "pytest", module, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ {module}: PASSED")
                results[module] = "PASSED"
            else:
                print(f"❌ {module}: FAILED")
                print(f"   Error: {result.stderr[:200]}...")
                results[module] = "FAILED"
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {module}: TIMEOUT")
            results[module] = "TIMEOUT"
        except Exception as e:
            print(f"💥 {module}: ERROR - {str(e)}")
            results[module] = "ERROR"
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 40)
    
    passed = sum(1 for status in results.values() if status == "PASSED")
    failed = sum(1 for status in results.values() if status == "FAILED")
    timeout = sum(1 for status in results.values() if status == "TIMEOUT")
    error = sum(1 for status in results.values() if status == "ERROR")
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏰ Timeout: {timeout}")
    print(f"💥 Error: {error}")
    print(f"📊 Total: {len(results)}")
    
    # Detailed results
    print(f"\n📝 Detailed Results:")
    print("-" * 40)
    
    for module, status in results.items():
        status_icon = {
            "PASSED": "✅",
            "FAILED": "❌", 
            "TIMEOUT": "⏰",
            "ERROR": "💥"
        }.get(status, "❓")
        
        print(f"{status_icon} {module}: {status}")
    
    # Overall status
    if failed == 0 and timeout == 0 and error == 0:
        print(f"\n🎉 ALL TESTS PASSED! Enhanced features are working correctly.")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
        return False

def run_quick_functionality_test():
    """Run a quick functionality test without pytest."""
    
    print("\n🔧 Quick Functionality Test")
    print("-" * 40)
    
    try:
        # Test imports
        print("Testing imports...")
        
        from advanced_indicators import AdvancedIndicatorEngine, add_advanced_indicators
        from risk_metrics import RiskAnalyzer
        from optimization import PortfolioOptimizer, ParameterOptimizer
        from benchmarking import BenchmarkAnalyzer, PerformanceAttributor
        
        print("✅ All enhanced modules imported successfully")
        
        # Test basic functionality
        print("Testing basic functionality...")
        
        # Create sample data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': 100 + np.random.randn(50),
            'high': 101 + np.random.randn(50),
            'low': 99 + np.random.randn(50),
            'close': 100 + np.random.randn(50),
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Test advanced indicators
        engine = AdvancedIndicatorEngine()
        result = engine.add_all_advanced_indicators(sample_data)
        print(f"✅ Advanced indicators: {len(result.columns)} columns")
        
        # Test risk analysis
        risk_analyzer = RiskAnalyzer()
        returns = sample_data['close'].pct_change().dropna()
        risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(returns)
        print(f"✅ Risk analysis: {len(risk_metrics)} metrics calculated")
        
        # Test portfolio optimization
        portfolio_returns = pd.DataFrame({
            'Asset1': np.random.normal(0.001, 0.02, 50),
            'Asset2': np.random.normal(0.0008, 0.018, 50)
        })
        
        optimizer = PortfolioOptimizer()
        opt_result = optimizer.mean_variance_optimization(portfolio_returns)
        print(f"✅ Portfolio optimization: {'Success' if opt_result['success'] else 'Failed'}")
        
        # Test benchmarking
        benchmark_analyzer = BenchmarkAnalyzer()
        benchmark_analyzer.add_benchmark('Test', portfolio_returns['Asset1'])
        print("✅ Benchmarking: Setup complete")
        
        print("\n🎉 Quick functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick functionality test failed: {e}")
        return False

def main():
    """Main test runner function."""
    
    print("🚀 StockStats Alpha Lab - Enhanced Features Test Runner")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('src') or not os.path.exists('tests'):
        print("❌ Error: Please run this script from the project root directory")
        return False
    
    # Run quick functionality test first
    if not run_quick_functionality_test():
        print("\n⚠️  Quick test failed. Skipping full test suite.")
        return False
    
    # Run full test suite
    success = run_test_suite()
    
    if success:
        print(f"\n🏆 ALL ENHANCED FEATURES ARE WORKING CORRECTLY!")
        print(f"📈 Ready for portfolio demonstration!")
    else:
        print(f"\n⚠️  Some tests failed. Please check the output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
