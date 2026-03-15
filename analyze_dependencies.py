"""
Dependency Analysis - Check which files are safe to delete
"""
from pathlib import Path
import re

def analyze_imports(file_path):
    """Extract all import statements from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all import statements
        imports = []
        
        # Standard imports: import module
        for match in re.finditer(r'^import\s+([^\s]+)', content, re.MULTILINE):
            imports.append(match.group(1))
        
        # From imports: from module import something
        for match in re.finditer(r'^from\s+([^\s]+)\s+import', content, re.MULTILINE):
            imports.append(match.group(1))
        
        return imports
    except:
        return []

def check_dependencies():
    """Check dependencies of all files"""
    workspace = Path(__file__).parent
    
    print("="*70)
    print("🔍 DEPENDENCY ANALYSIS".center(70))
    print("="*70)
    
    # Critical files that must work
    critical_files = [
        'src/main_multi_coin_dqn.py',
        'validate_model.py',
        'monitor_training_detailed.py',
        'watch_training.py',
        'resume_training.py',
    ]
    
    print("\n📌 CRITICAL FILES (Must keep their dependencies):\n")
    
    all_dependencies = set()
    
    for file_path in critical_files:
        full_path = workspace / file_path
        if full_path.exists():
            imports = analyze_imports(full_path)
            
            # Filter for src imports only
            src_imports = [imp for imp in imports if imp.startswith('src.')]
            
            print(f"✅ {file_path}")
            if src_imports:
                for imp in src_imports:
                    print(f"   └─ {imp}")
                    all_dependencies.add(imp)
            else:
                print(f"   └─ No src imports")
    
    print("\n" + "="*70)
    print("📦 REQUIRED MODULES")
    print("="*70)
    
    required_modules = set()
    for dep in all_dependencies:
        # Extract module path (e.g., src.models.dqn_agent -> src/models/dqn_agent.py)
        parts = dep.split('.')
        if len(parts) >= 2:
            module_file = '/'.join(parts) + '.py'
            required_modules.add(module_file)
    
    for mod in sorted(required_modules):
        print(f"  ✅ KEEP: {mod}")
    
    # Files to check for deletion
    check_files = {
        # Old training files
        'src/main_multi_coin.py': 'Old Q-Learning script',
        'src/run_test.py': 'Old test script',
        
        # Old model files
        'src/models/q_learning_gpu.py': 'Old Q-Learning (replaced by DQN)',
        'src/models/enhanced_training_gpu.py': 'Old training method',
        'src/models/trend_trading.py': 'Standalone trend trading',
        
        # Old data files
        'src/data/binance_data.py': 'Single source data fetcher',
        'src/data/download_eth_binance.py': 'Manual ETH download',
        'src/data/download_eth_kaggle.py': 'Manual ETH download',
        'src/data/fix_eth_hourly.py': 'One-time ETH fix',
        'src/data/kaggle_data.py': 'Kaggle data loader',
        'src/data/prepare_multi_coin_data.py': 'One-time data prep',
        
        # Old visualization files
        'src/visualization/binance_chart.py': 'Binance specific chart',
        'src/visualization/bitcoin_history.py': 'Bitcoin history viz',
        'src/visualization/trading_chart.py': 'Old chart method',
        
        # Root files
        'paper_content.txt': 'Extracted paper text',
        'PROJECT_STRUCTURE.txt': 'Old structure doc',
    }
    
    print("\n" + "="*70)
    print("🗑️  FILES SAFE TO DELETE")
    print("="*70)
    
    safe_to_delete = []
    
    for file_path, description in check_files.items():
        full_path = workspace / file_path
        
        # Check if file is in required modules
        is_required = file_path in required_modules
        
        if full_path.exists():
            if is_required:
                print(f"  ⚠️  KEEP: {file_path}")
                print(f"      └─ Reason: Used by critical files")
            else:
                print(f"  ✅ DELETE: {file_path}")
                print(f"      └─ {description}")
                safe_to_delete.append(file_path)
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"  Critical files analyzed: {len(critical_files)}")
    print(f"  Required modules: {len(required_modules)}")
    print(f"  Safe to delete: {len(safe_to_delete)}")
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    print("\n✅ DEFINITELY KEEP:")
    print("  - src/main_multi_coin_dqn.py (MAIN training)")
    print("  - src/models/dqn_agent.py")
    print("  - src/models/dqn_network.py")
    print("  - src/models/mdp_trading.py")
    print("  - src/models/replay_buffer.py")
    print("  - src/data/multi_coin_loader.py")
    print("  - src/utils/checkpoint.py")
    print("  - src/utils/indicators.py")
    print("  - All monitoring scripts (validate, monitor, watch)")
    
    print("\n🗑️  SAFE TO DELETE:")
    for file_path in safe_to_delete:
        print(f"  - {file_path}")
    
    print("\n⚠️  CONSIDER KEEPING (for future):")
    print("  - web/ directory (if you plan to use web interface)")
    print("  - tests/ directory (if you plan to add tests)")
    
    print("\n💾 To delete safely:")
    print("  1. Review the list above")
    print("  2. Backup important data first")
    print("  3. Run: python cleanup_safe.py --execute")
    
    print("\n" + "="*70)
    
    return safe_to_delete

if __name__ == "__main__":
    safe_files = check_dependencies()
