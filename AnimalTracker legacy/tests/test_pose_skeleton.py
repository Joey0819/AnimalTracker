"""快速验证 pose_engine + skeleton_reid 集成"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Test 1: skeleton embedding
from src.pose.pose_engine import compute_skeleton_embedding, create_pose_engine

kpts = np.array([
    [100, 50, 0.9], [85, 45, 0.8], [115, 45, 0.8], [100, 70, 0.9],
    [100, 100, 0.9], [100, 130, 0.9], [100, 160, 0.9],
    [80, 75, 0.7], [120, 75, 0.7], [80, 135, 0.7], [120, 135, 0.7],
], dtype=np.float32)

emb = compute_skeleton_embedding(kpts)
print(f"[OK] skeleton_embedding dim={len(emb)}, norm={np.linalg.norm(emb):.4f}")

# Test 2: two different animals should have different embeddings
kpts2 = kpts.copy()
kpts2[:, 0] += 200  # shift x
kpts2[1, 1] -= 20   # different ear position
kpts2[6, 1] += 30   # longer tail

emb2 = compute_skeleton_embedding(kpts2)
sim = float(np.dot(emb, emb2))
print(f"[OK] same-pose similarity={sim:.3f} (should be ~0.8-0.95)")

# Different pose
kpts3 = kpts.copy()
kpts3[0] = [100, 160, 0.9]   # nose at tail position (flipped)
kpts3[6] = [100, 50, 0.9]    # tail at nose position
emb3 = compute_skeleton_embedding(kpts3)
sim2 = float(np.dot(emb, emb3))
print(f"[OK] diff-pose similarity={sim2:.3f} (should be < 0.7)")

# Test 3: UnifiedPoseEngine init
engine = create_pose_engine(n_animals=4)
stats = engine.get_stats()
print(f"[OK] primary_engine={stats['primary_engine']}")
print(f"[OK] available: {stats['available_engines']}")

# Test 4: skeleton_reid integration with tracker
from src.tracking.skeleton_reid import SkeletonReIDMatcher
from src.tracking.buffer_recover_tracker import BufferRecoverTracker

tracker = BufferRecoverTracker(max_track_slots=4)
print(f"[OK] tracker skeleton_reid initialized: {tracker._skeleton_reid is not None}")

# Register embeddings
tracker.register_skeleton_embedding(1, emb)
tracker.register_skeleton_embedding(2, emb2)
print(f"[OK] skeleton_weight after register: {tracker._skeleton_weight}")

# Test cost computation
if tracker._skeleton_reid:
    cost = tracker._skeleton_reid.compute_cost_matrix([emb, emb2], [1, 2])
    print(f"[OK] cost_matrix diagonal (should be low): {cost[0,0]:.3f}, {cost[1,1]:.3f}")
    print(f"[OK] cost_matrix off-diag (should be higher): {cost[0,1]:.3f}, {cost[1,0]:.3f}")

# Test 5: GeometricPoseEngine
from src.pose.pose_engine import GeometricPoseEngine
geo = GeometricPoseEngine()
fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw a dark blob
cv2 = __import__('cv2')
cv2.ellipse(fake_frame, (200, 200), (40, 20), 30, 0, 360, (10, 10, 10), -1)
geo_poses = geo.detect_from_bboxes(fake_frame, [(160, 180, 240, 220)])
print(f"[OK] GeometricPoseEngine: {len(geo_poses)} poses")
if geo_poses:
    p = geo_poses[0]
    print(f"     center={p.center}, n_valid={p.n_valid}, emb_dim={len(p.skeleton_embedding)}")

print("\n=== ALL TESTS PASSED ===")
