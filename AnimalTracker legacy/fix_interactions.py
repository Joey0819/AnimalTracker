"""
修复扩展交互检测器的bug
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 修复 detect_interactions 方法
fix_code = '''
def detect_interactions(self,
                       tracks: Dict,
                       frame_idx: int,
                       velocities: Optional[Dict] = None) -> List[Dict]:
    """
    检测所有交互

    Args:
        tracks: 轨迹字典 {track_id: track}
        frame_idx: 当前帧号
        velocities: 速度字典 {track_id: velocity}

    Returns:
        interactions: 交互列表
    """
    interactions = []
    track_ids = list(tracks.keys())

    # 如果 velocities 为 None，创建空字典
    if velocities is None:
        velocities = {}

    # 计算所有轨迹对之间的交互
    for i, id1 in enumerate(track_ids):
        for id2 in track_ids[i+1:]:
            interaction = self._detect_pair_interaction(
                tracks[id1], tracks[id2],
                velocities.get(id1), velocities.get(id2),
                frame_idx
            )

            if interaction:
                interactions.append(interaction)

                # 记录历史
                self.interaction_history[(id1, id2)].append(interaction)

    return interactions
'''

# 读取文件
with open('C:/Users/cheng/.openclaw/workspace/AnimalTracker/src/behavior/extended_interactions.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到并替换方法
import re
pattern = r'def detect_interactions\(self,.*?\n(?=\n    def|\nclass|\Z)'
new_method = fix_code.strip()

content = re.sub(pattern, new_method, content, flags=re.DOTALL)

# 写回文件
with open('C:/Users/cheng/.openclaw/workspace/AnimalTracker/src/behavior/extended_interactions.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed extended_interactions.py")
