"""
Checkpoint utility for saving and loading training progress
Module hỗ trợ lưu và tải lại tiến trình training
"""

import pickle
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


class TrainingCheckpoint:
    """
    Class quản lý checkpoint cho training
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Khởi tạo checkpoint manager
        
        Args:
            checkpoint_dir: Thư mục lưu checkpoint
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, 
                       Q: Any,
                       episode: int,
                       history: Dict,
                       mdp_state: Optional[Dict] = None,
                       metadata: Optional[Dict] = None,
                       checkpoint_name: str = "checkpoint_latest.pkl"):
        """
        Lưu checkpoint
        
        Args:
            Q: Q-table hoặc Q-agent
            episode: Episode hiện tại
            history: Training history
            mdp_state: Trạng thái MDP (balance, holdings, etc.)
            metadata: Thông tin bổ sung
            checkpoint_name: Tên file checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # ✅ FIX PICKLE ERROR: Convert defaultdict to dict before pickling
        # If Q is an agent object with defaultdict Q-table, we need special handling
        agent_Q_original = None
        if hasattr(Q, 'Q'):
            # It's a QLearningAgent object with Q-table
            # Check if Q.Q is a defaultdict (can't pickle lambda)
            if hasattr(Q.Q, '__missing__'):  # It's a defaultdict
                # Save original, convert to dict temporarily
                agent_Q_original = Q.Q
                Q.Q = dict(agent_Q_original)
        
        # Also handle if Q itself is a defaultdict
        if isinstance(Q, dict) and hasattr(Q, '__missing__'):
            # Convert defaultdict to regular dict
            Q = dict(Q)
        
        checkpoint_data = {
            'episode': episode,
            'history': history,
            'Q': Q,
            'mdp_state': mdp_state,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # ✅ Restore original defaultdict after pickling
            if agent_Q_original is not None and hasattr(Q, 'Q'):
                Q.Q = agent_Q_original
            
            # Lưu metadata dạng JSON để dễ đọc
            meta_path = checkpoint_path.replace('.pkl', '_meta.json')
            with open(meta_path, 'w') as f:
                # ✅ Đọc total_episodes từ history['total_episodes'] hoặc episodes list
                total_episodes = history.get('total_episodes', 0)
                if total_episodes == 0 and history.get('episodes'):
                    total_episodes = history['episodes'][-1] if history['episodes'] else 0
                
                # ✅ Đọc best_profit từ history
                best_profit = history.get('best_profit', 0)
                if best_profit == 0 and history.get('rewards'):
                    best_profit = max(history['rewards']) if history['rewards'] else 0
                
                json.dump({
                    'episode': episode,
                    'timestamp': checkpoint_data['timestamp'],
                    'total_episodes': total_episodes,
                    'best_profit': best_profit,
                    'best_episode': history.get('best_episode', episode),
                    'metadata': metadata or {}
                }, f, indent=2)
            
            print(f"✅ Checkpoint saved: {checkpoint_path}")
            print(f"   Episode: {episode}")
            print(f"   Timestamp: {checkpoint_data['timestamp']}")
            
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_name: str = "checkpoint_latest.pkl") -> Optional[Dict]:
        """
        Load checkpoint
        
        Args:
            checkpoint_name: Tên file checkpoint
        
        Returns:
            Checkpoint data hoặc None nếu không tìm thấy
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return None
        
        # ✅ V4: Kiểm tra file size
        file_size = os.path.getsize(checkpoint_path)
        if file_size == 0:
            print(f"⚠️  Checkpoint file is empty: {checkpoint_path}")
            print(f"   💡 Training might be running or file is corrupt.")
            return None
        
        try:
            # ✅ V4: Retry mechanism cho file đang được ghi
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                    
                    print(f"✅ Checkpoint loaded: {checkpoint_path}")
                    print(f"   Episode: {checkpoint_data.get('episode', 'N/A')}")
                    print(f"   Timestamp: {checkpoint_data.get('timestamp', 'N/A')}")
                    if 'history' in checkpoint_data:
                        print(f"   Best profit: {checkpoint_data['history'].get('best_profit', 0):.4f}")
                    
                    return checkpoint_data
                    
                except (pickle.UnpicklingError, EOFError) as e:
                    if attempt < max_retries - 1:
                        import time
                        wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                        print(f"⚠️  Pickle error (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"   Waiting {wait_time}s and retrying... (file might be writing)")
                        time.sleep(wait_time)
                    else:
                        raise
                        
        except Exception as e:
            print(f"❌ Error loading checkpoint after {max_retries} attempts: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def list_checkpoints(self) -> list:
        """
        Liệt kê tất cả checkpoints
        
        Returns:
            List các checkpoint files
        """
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')]
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Xóa checkpoint
        
        Args:
            checkpoint_name: Tên file checkpoint
        
        Returns:
            True nếu xóa thành công
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        meta_path = checkpoint_path.replace('.pkl', '_meta.json')
        
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            print(f"✅ Checkpoint deleted: {checkpoint_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting checkpoint: {e}")
            return False
    
    def auto_save_checkpoint(self,
                            Q: Any,
                            episode: int,
                            history: Dict,
                            mdp_state: Optional[Dict] = None,
                            metadata: Optional[Dict] = None,
                            save_interval: int = 100,
                            keep_last_n: int = 5):
        """
        Auto-save checkpoint mỗi N episodes và giữ lại N checkpoints gần nhất
        
        Args:
            Q: Q-table hoặc Q-agent
            episode: Episode hiện tại
            history: Training history
            mdp_state: Trạng thái MDP
            metadata: Thông tin bổ sung
            save_interval: Lưu mỗi N episodes
            keep_last_n: Giữ lại N checkpoints gần nhất
        """
        # Chỉ save khi đến interval
        if episode % save_interval != 0 and episode != 0:
            return
        
        # Tạo tên checkpoint với episode number
        checkpoint_name = f"checkpoint_ep{episode:05d}.pkl"
        
        # Save checkpoint
        self.save_checkpoint(Q, episode, history, mdp_state, metadata, checkpoint_name)
        
        # Cũng save checkpoint_latest để dễ load
        self.save_checkpoint(Q, episode, history, mdp_state, metadata, "checkpoint_latest.pkl")
        
        # Cleanup old checkpoints (giữ lại N gần nhất)
        self._cleanup_old_checkpoints(keep_last_n)
    
    def _cleanup_old_checkpoints(self, keep_last_n: int):
        """
        Xóa các checkpoints cũ, chỉ giữ lại N gần nhất
        
        Args:
            keep_last_n: Số lượng checkpoints giữ lại
        """
        try:
            checkpoints = self.list_checkpoints()
            
            # Loại bỏ checkpoint_latest khỏi danh sách cleanup
            checkpoints = [c for c in checkpoints if c != "checkpoint_latest.pkl"]
            
            # ✅ Sort theo tên để đảm bảo thứ tự (episode number)
            # Format: checkpoint_ep00010.pkl
            def extract_episode_num(name):
                try:
                    # Tìm số episode từ tên file
                    if 'ep' in name:
                        parts = name.split('ep')
                        if len(parts) > 1:
                            num_str = parts[1].split('.')[0]
                            return int(num_str)
                    return 0
                except:
                    return 0
            
            checkpoints.sort(key=extract_episode_num, reverse=True)  # Mới nhất trước
            
            # Xóa các checkpoint cũ (giữ lại N gần nhất)
            if len(checkpoints) > keep_last_n:
                old_checkpoints = checkpoints[keep_last_n:]
                for checkpoint in old_checkpoints:
                    self.delete_checkpoint(checkpoint)
                    # Chỉ log nếu xóa thành công
        except Exception as e:
            # ✅ Không throw exception để tránh dừng training
            print(f"⚠️ Warning: Error cleaning up old checkpoints: {e}")


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo checkpoint manager
    checkpoint = TrainingCheckpoint("checkpoints")
    
    # Giả lập Q-table và history
    from collections import defaultdict
    import numpy as np
    
    Q = defaultdict(lambda: np.zeros(3))
    Q[(0, 1, 2)] = np.array([0.5, 0.3, 0.8])
    
    history = {
        'episodes': [1, 2, 3],
        'profits': [0.1, 0.2, 0.15],
        'best_profit': 0.2
    }
    
    # Save checkpoint
    print("=== Saving checkpoint ===")
    checkpoint.save_checkpoint(Q, episode=3, history=history)
    
    # List checkpoints
    print("\n=== List checkpoints ===")
    checkpoints = checkpoint.list_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp}")
    
    # Load checkpoint
    print("\n=== Loading checkpoint ===")
    data = checkpoint.load_checkpoint()
    if data:
        print(f"Loaded episode: {data['episode']}")
        print(f"History length: {len(data['history']['episodes'])}")
