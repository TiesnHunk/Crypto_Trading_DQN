"""
Cleanup script - Remove old/unused files
"""
import os
from pathlib import Path

# Files to KEEP
KEEP_FILES = {
    # Core documentation
    'README.md',
    'V6.1_REMOVE_STOP_LOSS.md',
    'V6_PAPER_COMPLIANT_UPDATE.md',
    'PAPER_ANALYSIS_MDD.md',
    'FIX_MDD_PENALTY_ISSUE.md',
    
    # Core training scripts (CRITICAL - DO NOT DELETE!)
    'validate_model.py',
    'monitor_training_detailed.py',
    'watch_training.py',
    'resume_training.py',
    
    # Core utilities
    'check_multi_coin_data.py',
    'read_paper.py',
    'cleanup_old_files.py',
}

# Source code directories to KEEP (DO NOT TOUCH!)
KEEP_DIRECTORIES = {
    'src',  # All training code: main_multi_coin_dqn.py, models/, etc.
    'data',  # All data files
    'logs',  # Training logs
    'paper',  # Research paper
}

# Python files to DELETE (old training/debug scripts)
OLD_PY_FILES = [
    'watch_training_v6.2.py',  # Old version
    'check_checkpoint.py',  # Replaced by quick_check_checkpoint.py
    'simple_gpu_monitor.py',  # Not needed anymore
    'monitor_gpu_training.py',  # Not needed anymore
    'test_dqn_one_episode.py',  # Old test
    'analyze_eth.py',  # Single coin analysis
    'visualize_state_collapse.py',  # Old debugging
    'validate_time_series.py',  # Replaced by validate_model.py
    'validate_with_realtime_data.py',  # Old validation
    'debug_action_selection.py',  # Old debugging
    'SAU_KHI_TRAIN_XONG.py',  # Old script
    'fix_qnetwork_and_retrain.py',  # Old fix
    'retrain_longer.py',  # Old training
    'post_training_analysis.py',  # Replaced by validate_model.py
    'retrain_model.py',  # Old training
    'check_performance_1000_episodes.py',  # Old check
    'monitor_training.py',  # Replaced by monitor_training_detailed.py
    'check_q_table.py',  # Q-learning specific
    'visualize_trading.py',  # Old visualization
    'quick_check_checkpoint.py',  # Debug only
    'test_checkpoint_after_fix.py',  # Debug only
    'debug_checkpoint.py',  # Debug only
]

# Markdown files to DELETE (old documentation)
OLD_MD_FILES = [
    'V5_MDD_ANNUALIZED_RETURN_UPDATE.md',  # Old version
    'CHECK_GPU_COMMANDS.md',
    'HUONG_DAN_CHECKPOINT_RESUME.md',
    'HUONG_DAN_MONITOR_GPU.md',
    'DQN_IMPLEMENTATION_STATUS.md',
    'TOM_TAT_VALIDATION_VA_QUYET_DINH.md',
    'PHAN_TICH_STATE_COLLAPSE.md',
    'HUONG_DAN_DOWNLOAD_DATA.md',
    'THAY_DOI_TIME_SERIES_SPLIT.md',
    'DANH_GIA_MODEL_STATUS.md',
    'DANH_GIA_OUTPUT_FORMAT.md',
    'FIX_REALTIME_OUTPUT.md',
    'FIX_CHECKPOINT_PICKLE.md',
    'LOG_UPDATE_SUMMARY.md',
    'UPDATE_LOG_EVERY_EPISODE.md',
    'STATUS_HIEN_TAI.md',
    'HUONG_DAN_TRAIN_CPU_GPU.md',
    'GPU_TRAINING_FIX.md',
    'COMPARISON_WITH_PAPER.md',
    'THEO_BAI_BAO_REWARD.md',
    'TOM_TAT_CAI_THIEN.md',
    'FIX_OVER_TRADING.md',
    'CRITICAL_ISSUE_SUMMARY.md',
    'MAJOR_FIXES_APPLIED.md',
    'README_MAJOR_RETRAIN.md',
    'HUONG_DAN_CAI_THIEN.md',
    'PHAN_TICH_KET_QUA_VALIDATION.md',
    'KET_QUA_TRAINING_15000.md',
    'TOM_TAT_KET_QUA_CUOI_CUNG.md',
    'FIXES_APPLIED.md',
    'README_CURRENT.md',
    'BAO_CAO_TRAINING.md',
    'TRAINING_ISSUE_NOTE.md',
    'HUONG_DAN_KIEM_TRA.md',
    'README_HIEN_TAI.md',
    'VAN_DE_CHUA_DUOC_FIX.md',
    'FINAL_STATUS.md',
    'UPDATE_TO_NEW_FILES.md',
    'PHAN_TICH_OUTPUT_TRAINING.md',
    'KET_LUAN_FIX.md',
    'README_FIX_APPLIED.md',
    'PHAN_TICH_CHI_TIET_VAN_DE.md',
    'FIX_ROOT_CAUSE.md',
    'README_TRAINING.md',
    'README_CURRENT_STATUS.md',
    'HUONG_DAN_TRAIN.md',
    'QUYET_DINH_STRATEGY.md',
    'TOM_TAT_SO_SANH.md',
    'SO_SANH_STRATEGIES.md',
    'TONG_KET_FINAL.md',
    'SAN_SANG_TRAIN.md',
    'TOM_TAT_STRATEGY.md',
    'HUONG_DAN_STRATEGY_MOI.md',
    'PHAN_TICH_STRATEGY_MOI.md',
    'KET_QUA_UPDATE_SOL_ADA.md',
    'KET_LUAN_STRATEGY.md',
    'TOM_TAT_TRAIN_VALIDATE.md',
    'PHAN_TICH_TRAIN_VALIDATE_COINS.md',
    'TOM_TAT_DANH_GIA.md',
    'DANH_GIA_YEU_CAU.md',
    'CHECKLIST_YEU_CAU_FINAL.md',
    'PHAN_TICH_KET_QUA_TRAINING.md',
    'TOM_TAT_PHAN_TICH.md',
    'HUONG_DAN_TRAIN_LAI.md',
    'SAU_KHI_TAI_DATA.md',
    'TOM_TAT_SAU_TAI_DATA.md',
    'HUONG_DAN_TAI_LAI_DATA.md',
    'TOM_TAT_XOA_CSV.md',
]

def cleanup_files(dry_run=True):
    """Delete old files"""
    workspace = Path(__file__).parent
    
    deleted_count = 0
    kept_count = 0
    
    print("="*70)
    print("🗑️  CLEANUP OLD FILES".center(70))
    print("="*70)
    
    if dry_run:
        print("⚠️  DRY RUN MODE - No files will be deleted")
        print("   Run with --execute to actually delete files\n")
    else:
        print("🚨 EXECUTING - Files will be DELETED!\n")
    
    # Delete old Python files
    print("📄 Python files to delete:")
    for filename in OLD_PY_FILES:
        filepath = workspace / filename
        if filepath.exists():
            if dry_run:
                print(f"   ❌ Would delete: {filename}")
            else:
                filepath.unlink()
                print(f"   ✅ Deleted: {filename}")
            deleted_count += 1
        else:
            print(f"   ⚠️  Not found: {filename}")
    
    # Delete old Markdown files
    print(f"\n📝 Markdown files to delete ({len(OLD_MD_FILES)} files):")
    for filename in OLD_MD_FILES:
        filepath = workspace / filename
        if filepath.exists():
            if dry_run:
                print(f"   ❌ Would delete: {filename}")
            else:
                filepath.unlink()
                print(f"   ✅ Deleted: {filename}")
            deleted_count += 1
    
    # List files to keep
    print("\n✅ Files to KEEP:")
    for filename in KEEP_FILES:
        filepath = workspace / filename
        if filepath.exists():
            print(f"   ✓ {filename}")
            kept_count += 1
    
    # List directories to keep
    print("\n📁 Directories to KEEP (entire folder):")
    for dirname in KEEP_DIRECTORIES:
        dirpath = workspace / dirname
        if dirpath.exists():
            print(f"   ✓ {dirname}/ (ALL FILES INSIDE)")
            # Count files in directory
            file_count = sum(1 for _ in dirpath.rglob('*') if _.is_file())
            print(f"      └─ {file_count} files")
            kept_count += file_count
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Files to delete: {deleted_count}")
    print(f"Files to keep:   {kept_count}")
    
    if dry_run:
        print("\n💡 Run with --execute to actually delete files:")
        print("   python cleanup_old_files.py --execute")
    else:
        print("\n✅ Cleanup complete!")
    
    print("="*70)

if __name__ == "__main__":
    import sys
    
    # Check if --execute flag is provided
    execute = '--execute' in sys.argv
    
    cleanup_files(dry_run=not execute)
