"""
Safe Cleanup - Delete only verified safe files
Based on dependency analysis
"""
from pathlib import Path

# Files verified as SAFE TO DELETE (not used by critical scripts)
SAFE_TO_DELETE = [
    # Old training scripts
    'src/main_multi_coin.py',  # Old Q-Learning, replaced by main_multi_coin_dqn.py
    'src/run_test.py',  # Old test script
    
    # Old model files (replaced by DQN)
    'src/models/q_learning_gpu.py',  # Old Q-Learning implementation
    'src/models/enhanced_training_gpu.py',  # Old training method
    'src/models/trend_trading.py',  # Standalone trend (integrated into MDP)
    
    # One-time data download scripts (already executed)
    'src/data/download_eth_binance.py',  # Manual ETH download (done)
    'src/data/download_eth_kaggle.py',  # Manual ETH download (done)
    'src/data/fix_eth_hourly.py',  # One-time fix (done)
    
    # Old visualization (not used)
    'src/visualization/binance_chart.py',  # Single coin chart
    'src/visualization/bitcoin_history.py',  # Bitcoin specific
    'src/visualization/trading_chart.py',  # Old chart method
    
    # Old documentation
    'paper_content.txt',  # Extracted text (can re-extract)
    'PROJECT_STRUCTURE.txt',  # Old structure (replaced by HUONG_DAN_SU_DUNG.md)
]

# Files to KEEP (might be needed later)
KEEP_FOR_FUTURE = [
    'src/data/binance_data.py',  # Might need for downloading new data
    'src/data/kaggle_data.py',  # Alternative data source
    'src/data/prepare_multi_coin_data.py',  # Re-download & prepare data when needed
]

def cleanup_safe(dry_run=True):
    """Safely delete verified unused files"""
    workspace = Path(__file__).parent
    
    print("="*70)
    print("🗑️  SAFE CLEANUP - DELETE VERIFIED UNUSED FILES".center(70))
    print("="*70)
    
    if dry_run:
        print("⚠️  DRY RUN MODE - No files will be deleted")
        print("   Run with --execute to actually delete\n")
    else:
        print("🚨 EXECUTING - Files will be DELETED!")
        print("⚠️  Last chance to Ctrl+C!\n")
        import time
        for i in range(3, 0, -1):
            print(f"   Deleting in {i}...")
            time.sleep(1)
        print()
    
    deleted = []
    not_found = []
    total_size = 0
    
    for file_path in SAFE_TO_DELETE:
        full_path = workspace / file_path
        
        if full_path.exists():
            file_size = full_path.stat().st_size
            size_kb = file_size / 1024
            total_size += file_size
            
            if dry_run:
                print(f"   ❌ Would delete: {file_path} ({size_kb:.1f} KB)")
            else:
                full_path.unlink()
                print(f"   ✅ Deleted: {file_path} ({size_kb:.1f} KB)")
            
            deleted.append(file_path)
        else:
            not_found.append(file_path)
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"  Files deleted:   {len(deleted)}")
    print(f"  Files not found: {len(not_found)}")
    print(f"  Space freed:     {total_size / 1024:.1f} KB")
    
    if not_found:
        print(f"\n📝 Files not found (already deleted?):")
        for file_path in not_found:
            print(f"  - {file_path}")
    
    print("\n✅ FILES KEPT (still needed):")
    print("  ✓ src/main_multi_coin_dqn.py (MAIN training)")
    print("  ✓ src/models/dqn_agent.py")
    print("  ✓ src/models/dqn_network.py")
    print("  ✓ src/models/mdp_trading.py")
    print("  ✓ src/models/replay_buffer.py")
    print("  ✓ src/models/metrics.py")
    print("  ✓ src/data/multi_coin_loader.py")
    print("  ✓ src/data/binance_data.py (for future data updates)")
    print("  ✓ src/data/kaggle_data.py (alternative data source)")
    print("  ✓ src/data/prepare_multi_coin_data.py (re-download data)")
    print("  ✓ src/utils/checkpoint.py")
    print("  ✓ src/utils/indicators.py")
    print("  ✓ src/config/config.py")
    print("  ✓ All monitoring scripts")
    
    print("\n💡 FILES KEPT FOR FUTURE USE:")
    for file_path in KEEP_FOR_FUTURE:
        print(f"  ⚠️  {file_path}")
    print("     └─ Might need when downloading new data")
    
    if dry_run:
        print("\n💡 To execute cleanup:")
        print("   python cleanup_safe.py --execute")
    else:
        print("\n✅ Cleanup complete!")
        print("💾 Checkpoint: Run validate_model.py to verify training still works")
    
    print("="*70)

if __name__ == "__main__":
    import sys
    
    execute = '--execute' in sys.argv
    cleanup_safe(dry_run=not execute)
