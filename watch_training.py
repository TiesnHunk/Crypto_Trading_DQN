"""
Training Progress Monitor
Continuously displays GPU stats and reads actual training progress from checkpoint
"""

import subprocess
import time
from datetime import datetime, timedelta
import os
from pathlib import Path
import torch

class TrainingMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.training_detected_at = None
        self.last_gpu_active = None
        self.checkpoint_dir = Path("src/checkpoints_dqn")
        
    def get_checkpoint_info(self):
        """Read actual episode from checkpoint file"""
        checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pkl"
        
        if not checkpoint_path.exists():
            return None
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            episode = checkpoint.get('episode', None)
            epsilon = checkpoint.get('epsilon', None)
            meta = checkpoint.get('metadata', {})
            
            return {
                'episode': episode,
                'epsilon': epsilon,
                'best_profit': meta.get('best_profit', 0),
                'training_time': meta.get('training_time', 0)
            }
        except Exception:
            return None
        
    def get_gpu_quick(self):
        """Quick GPU check"""
        nvidia_paths = ['nvidia-smi', r'C:\Windows\System32\nvidia-smi.exe']
        
        for path in nvidia_paths:
            try:
                result = subprocess.run(
                    [path, '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    vals = result.stdout.strip().split(', ')
                    return {
                        'util': float(vals[0]),
                        'mem': float(vals[1]),
                        'temp': float(vals[2])
                    }
            except:
                continue
        return None
    
    def estimate_progress(self, gpu):
        """Get actual training progress from checkpoint file"""
        if not gpu:
            return None
            
        # Training detected if GPU util > 15%
        if gpu['util'] > 15:
            if self.training_detected_at is None:
                self.training_detected_at = datetime.now()
            self.last_gpu_active = datetime.now()
            
            # Try to read actual episode from checkpoint
            checkpoint_info = self.get_checkpoint_info()
            
            if checkpoint_info and checkpoint_info['episode']:
                episode = checkpoint_info['episode']
                
                # Progress percentage
                progress = (episode / 5000) * 100
                
                # Use actual training time from checkpoint
                training_time = checkpoint_info['training_time']
                avg_time_per_episode = training_time / episode if episode > 0 else 150
                
                # Remaining time calculation
                remaining_episodes = 5000 - episode
                # Adjust for epsilon decay (episodes get slightly faster)
                # At epsilon 0.3, episodes ~10% faster than at epsilon 1.0
                epsilon_factor = max(0.85, checkpoint_info.get('epsilon', 0.5))
                remaining_seconds = remaining_episodes * avg_time_per_episode * epsilon_factor
                eta = datetime.now() + timedelta(seconds=remaining_seconds)
                
                return {
                    'episode': episode,
                    'progress': progress,
                    'eta': eta,
                    'elapsed': training_time,
                    'epsilon': checkpoint_info['epsilon'],
                    'best_profit': checkpoint_info['best_profit'],
                    'avg_time_per_ep': avg_time_per_episode,
                    'remaining_eps': remaining_episodes,
                    'remaining_secs': remaining_seconds
                }
            else:
                # Fallback: estimate based on elapsed time
                elapsed = (datetime.now() - self.training_detected_at).total_seconds()
                estimated_episode = max(1, int(elapsed / 180))  # 180 sec for early episodes
                
                return {
                    'episode': estimated_episode,
                    'progress': (estimated_episode / 5000) * 100,
                    'eta': None,
                    'elapsed': elapsed,
                    'epsilon': None,
                    'best_profit': None
                }
        
        return None
    
    def format_time(self, seconds):
        """Format seconds to readable string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(seconds / 86400)
            hours = int((seconds % 86400) / 3600)
            return f"{days}d {hours}h"
    
    def display_status(self, gpu, progress):
        """Display current status"""
        os.system('cls')
        
        print("\n" + "="*70)
        print("🎮 DQN TRAINING MONITOR".center(70))
        print("="*70)
        print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Monitor Running: {self.format_time((datetime.now() - self.start_time).total_seconds())}")
        print("="*70)
        
        # GPU Status
        print("\n📊 GPU STATUS:")
        if gpu:
            util_bar = "█" * int(gpu['util']/2.5) + "░" * (40 - int(gpu['util']/2.5))
            mem_bar = "█" * int(gpu['mem']/100) + "░" * (40 - int(gpu['mem']/100))
            
            status_icon = "🟢" if gpu['util'] > 20 else "🟡" if gpu['util'] > 5 else "🔴"
            
            print(f"  {status_icon} Utilization: [{util_bar}] {gpu['util']:.1f}%")
            print(f"  💾 Memory:      [{mem_bar}] {gpu['mem']:.0f} MiB")
            print(f"  🌡️  Temperature: {gpu['temp']:.0f}°C")
        else:
            print("  ❌ GPU not detected")
        
        # Training Progress
        print("\n" + "="*70)
        print("🚀 TRAINING PROGRESS:")
        
        if progress and progress['episode'] > 0:
            episode = progress['episode']
            prog_bar = "█" * int(progress['progress']/2.5) + "░" * (40 - int(progress['progress']/2.5))
            
            print(f"  📈 Episode:     {episode:,} / 5,000")
            print(f"  📊 Progress:    [{prog_bar}] {progress['progress']:.2f}%")
            print(f"  ⏱️  Elapsed:     {self.format_time(progress['elapsed'])}")
            
            # Show epsilon if available
            if progress.get('epsilon'):
                print(f"  🎲 Epsilon:     {progress['epsilon']:.6f}")
            
            # Show best profit if available  
            if progress.get('best_profit'):
                print(f"  💰 Best Profit: ${progress['best_profit']:.2f}")
            
            # DEBUG: Show calculation details
            if progress.get('avg_time_per_ep'):
                print(f"  📐 Avg Time/Ep: {progress['avg_time_per_ep']:.1f}s")
            if progress.get('remaining_eps'):
                print(f"  📊 Remaining:   {progress['remaining_eps']:,} episodes")
            
            # Show ETA if available
            if progress.get('eta'):
                print(f"  🕐 ETA:         {progress['eta'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Time remaining (ensure it's positive)
                remaining = max(0, (progress['eta'] - datetime.now()).total_seconds())
                if remaining > 0:
                    print(f"  ⏳ Remaining:   {self.format_time(remaining)}")
                else:
                    print(f"  ⏳ Remaining:   Calculating... (episode {episode}/5000)")
            else:
                print("  ⏳ Status:      Early episodes (calculating ETA...)")
        elif self.training_detected_at:
            elapsed = (datetime.now() - self.training_detected_at).total_seconds()
            print(f"  ⏳ Training started {self.format_time(elapsed)} ago")
            print("  📊 Loading checkpoint data...")
        else:
            print("  ⏸️  Waiting for training to start...")
            print("  💡 Training should show GPU util > 20%")
        
        # Footer
        print("\n" + "="*70)
        print("🔄 Auto-refreshing every 5 seconds | Press Ctrl+C to stop")
        print("="*70)
    
    def run(self):
        """Main monitoring loop"""
        print("\n🚀 Starting Training Monitor...")
        print("📊 Monitoring DQN Multi-Coin Training")
        print("🔄 Refresh interval: 5 seconds\n")
        time.sleep(2)
        
        try:
            while True:
                gpu = self.get_gpu_quick()
                progress = self.estimate_progress(gpu)
                self.display_status(gpu, progress)
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitor stopped")
            
            if self.training_detected_at:
                elapsed = (datetime.now() - self.training_detected_at).total_seconds()
                estimated_episodes = int(elapsed / 148)
                print(f"📊 Training ran for: {self.format_time(elapsed)}")
                print(f"🎯 Estimated episodes: {estimated_episodes:,} / 5,000")


if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.run()
