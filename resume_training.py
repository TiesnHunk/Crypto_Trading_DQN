"""
Resume Training Script - Continue DQN training from last checkpoint
Automatically detects last episode and continues from there
"""

import sys
import torch
from pathlib import Path

def check_latest_checkpoint(checkpoint_dir="src/checkpoints_dqn"):
    """Check the latest checkpoint and show training status"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / 'checkpoint_latest.pkl'
    
    print("\n" + "="*70)
    print("🔍 CHECKPOINT STATUS CHECK")
    print("="*70)
    
    if not checkpoint_path.exists():
        print("\n❌ No checkpoint found!")
        print(f"   Looking for: {checkpoint_path}")
        print("\n💡 This means:")
        print("   - Training has not started yet, OR")
        print("   - Checkpoint was deleted")
        print("\n🚀 Run: python src/main_multi_coin_dqn.py")
        print("   (Will start training from Episode 1)")
        return None
    
    # Load checkpoint
    print(f"\n✅ Checkpoint found: {checkpoint_path.name}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        episode = checkpoint.get('episode', 'Unknown')
        epsilon = checkpoint.get('epsilon', 'Unknown')
        
        # Get metadata
        meta = checkpoint.get('metadata', {})
        all_episodes = meta.get('all_episodes', [])
        all_profits = meta.get('all_profits', [])
        best_profit = meta.get('best_profit', 0)
        best_episode = meta.get('best_episode', 0)
        training_time = meta.get('training_time', 0)
        
        print(f"\n📊 TRAINING PROGRESS:")
        print(f"   Last Episode: {episode:,} / 5,000")
        if episode != 'Unknown':
            progress = (episode / 5000) * 100
            bar = "█" * int(progress/2.5) + "░" * (40 - int(progress/2.5))
            print(f"   Progress:     [{bar}] {progress:.2f}%")
        print(f"   Epsilon:      {epsilon if isinstance(epsilon, str) else f'{epsilon:.6f}'}")
        
        if len(all_episodes) > 0:
            print(f"\n💰 PROFIT HISTORY:")
            recent_profits = all_profits[-10:] if len(all_profits) >= 10 else all_profits
            avg_recent = sum(recent_profits) / len(recent_profits) if recent_profits else 0
            print(f"   Last 10 episodes avg: ${avg_recent:.2f}")
            print(f"   Best profit: ${best_profit:.2f} (Episode {best_episode})")
        
        if training_time > 0:
            print(f"\n⏱️  TIMING:")
            print(f"   Training time: {training_time/60:.1f} min ({training_time/3600:.2f} hours)")
            avg_time_per_episode = training_time / episode if episode > 0 else 0
            print(f"   Avg per episode: {avg_time_per_episode:.2f} seconds")
            
            # Estimate remaining time
            remaining_episodes = 5000 - episode
            estimated_remaining = remaining_episodes * avg_time_per_episode
            print(f"   Estimated remaining: {estimated_remaining/3600:.1f} hours ({estimated_remaining/86400:.1f} days)")
        
        # Buffer info
        replay_buffer = checkpoint.get('replay_buffer', {})
        buffer_size = len(replay_buffer.get('buffer', []))
        buffer_capacity = replay_buffer.get('capacity', 100000)
        
        print(f"\n💾 REPLAY BUFFER:")
        print(f"   Size: {buffer_size:,} / {buffer_capacity:,} ({buffer_size/buffer_capacity*100:.1f}%)")
        
        # File info
        file_size = checkpoint_path.stat().st_size
        modified = checkpoint_path.stat().st_mtime
        from datetime import datetime
        modified_str = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n📁 FILE INFO:")
        print(f"   Size: {file_size/1024:.2f} KB")
        print(f"   Last saved: {modified_str}")
        
        print("\n" + "="*70)
        print("✅ READY TO RESUME")
        print("="*70)
        print("\n🚀 To continue training, run:")
        print("   python src/main_multi_coin_dqn.py")
        print("\n💡 Training will automatically resume from Episode", episode + 1 if isinstance(episode, int) else "?")
        print("   (resume=True is already set in the script)")
        print("\n" + "="*70 + "\n")
        
        return checkpoint
        
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        print("\n💡 Checkpoint may be corrupted. Options:")
        print("   1. Delete checkpoint and start fresh")
        print("   2. Try loading checkpoint_best.pkl instead")
        return None


def compare_all_checkpoints(checkpoint_dir="src/checkpoints_dqn"):
    """Compare all available checkpoints"""
    
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"\n❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoint_files = sorted(checkpoint_dir.glob("*.pkl"))
    
    if not checkpoint_files:
        print(f"\n❌ No checkpoints found in {checkpoint_dir}")
        return
    
    print("\n" + "="*70)
    print("📊 ALL CHECKPOINTS")
    print("="*70)
    
    for ckpt_file in checkpoint_files:
        print(f"\n📄 {ckpt_file.name}:")
        
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            
            episode = checkpoint.get('episode', 'N/A')
            epsilon = checkpoint.get('epsilon', 'N/A')
            meta = checkpoint.get('metadata', {})
            best_profit = meta.get('best_profit', 'N/A')
            
            print(f"   Episode: {episode}")
            print(f"   Epsilon: {epsilon if isinstance(epsilon, str) else f'{epsilon:.6f}'}")
            if best_profit != 'N/A':
                print(f"   Best Profit: ${best_profit:.2f}")
            
            file_size = ckpt_file.stat().st_size
            modified = ckpt_file.stat().st_mtime
            from datetime import datetime
            modified_str = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"   Size: {file_size/1024:.2f} KB")
            print(f"   Modified: {modified_str}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Show all checkpoints
        compare_all_checkpoints()
    else:
        # Check latest checkpoint and show resume info
        check_latest_checkpoint()
    
    print("\n💡 USAGE:")
    print("   python resume_training.py           # Check latest checkpoint")
    print("   python resume_training.py --all     # Compare all checkpoints")
    print()
