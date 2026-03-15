"""
Cleanup Phase 2 - Remove remaining old files
"""
from pathlib import Path

# Additional files to DELETE
OLD_FILES_PHASE2 = [
    # Old text files
    'paper_content.txt',  # Extract from paper, không cần thiết
    'PROJECT_STRUCTURE.txt',  # Old structure, thay bằng HUONG_DAN_SU_DUNG.md
    
    # Old Python files in src/
    'src/main_multi_coin.py',  # Old version, dùng main_multi_coin_dqn.py
    'src/run_test.py',  # Old test script
    
    # Old Python files in src/models/
    'src/models/q_learning_gpu.py',  # Q-learning old version, dùng DQN
    'src/models/enhanced_training_gpu.py',  # Old training method
    'src/models/trend_trading.py',  # Separate trend trading, integrated vào MDP
    
    # Old Python files in src/data/
    'src/data/binance_data.py',  # Single source, dùng multi_coin_loader
    'src/data/download_eth_binance.py',  # Manual download, không cần
    'src/data/download_eth_kaggle.py',  # Manual download, không cần
    'src/data/fix_eth_hourly.py',  # One-time fix, đã xong
    'src/data/kaggle_data.py',  # Kaggle specific, không dùng
    'src/data/prepare_multi_coin_data.py',  # One-time preparation, đã xong
    
    # Old Python files in src/visualization/
    'src/visualization/binance_chart.py',  # Single chart, không cần
    'src/visualization/bitcoin_history.py',  # Bitcoin specific
    'src/visualization/trading_chart.py',  # Old chart method
    
    # Web app (nếu không dùng)
    'web/app.py',  # Web interface, chưa cần thiết cho training
    'web/test_model.py',  # Web testing
    'web/start.bat',  # Web startup
    'web/README.md',  # Web docs
    'web/requirements.txt',  # Web dependencies
    
    # Tests (nếu chưa implement)
    'tests/setup.py',  # Test setup
]

def cleanup_phase2(dry_run=True):
    """Phase 2 cleanup"""
    workspace = Path(__file__).parent
    
    print("="*70)
    print("🗑️  CLEANUP PHASE 2 - REMAINING OLD FILES".center(70))
    print("="*70)
    
    if dry_run:
        print("⚠️  DRY RUN MODE - No files will be deleted")
        print("   Run with --execute to actually delete files\n")
    else:
        print("🚨 EXECUTING - Files will be DELETED!\n")
    
    deleted_count = 0
    not_found_count = 0
    
    # Check each file
    print("📋 Files to check:\n")
    
    for filename in OLD_FILES_PHASE2:
        filepath = workspace / filename
        
        if filepath.exists():
            file_size = filepath.stat().st_size
            size_kb = file_size / 1024
            
            if dry_run:
                print(f"   ❌ Would delete: {filename} ({size_kb:.1f} KB)")
            else:
                filepath.unlink()
                print(f"   ✅ Deleted: {filename} ({size_kb:.1f} KB)")
            deleted_count += 1
        else:
            not_found_count += 1
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Files found:     {deleted_count}")
    print(f"Files not found: {not_found_count}")
    
    if dry_run:
        print("\n💡 Files to DELETE:")
        print("   📄 Text files: paper_content.txt, PROJECT_STRUCTURE.txt")
        print("   🐍 Old Python: main_multi_coin.py, q_learning_gpu.py, etc.")
        print("   🌐 Web app: web/ directory (if not using)")
        print("   🧪 Tests: tests/ directory (if not implemented)")
        print("\n💡 Files to KEEP:")
        print("   ✅ main_multi_coin_dqn.py (MAIN training)")
        print("   ✅ dqn_agent.py, mdp_trading.py (CORE models)")
        print("   ✅ multi_coin_loader.py (DATA loader)")
        print("   ✅ All monitoring scripts")
        print("\n⚠️  RECOMMENDATION:")
        print("   Consider keeping web/ and tests/ for future use")
        print("   Only delete if you're sure you won't need them")
        print("\n💡 Run with --execute to delete:")
        print("   python cleanup_phase2.py --execute")
    else:
        print("\n✅ Cleanup Phase 2 complete!")
    
    print("="*70)

if __name__ == "__main__":
    import sys
    
    execute = '--execute' in sys.argv
    cleanup_phase2(dry_run=not execute)
