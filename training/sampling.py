from __future__ import annotations
import random
from collections import defaultdict

def scene_balanced_order(records: list[dict], seed: int) -> list[int]:
    """Interleave scenes so every prefix covers the available scene domain."""
    by_scene: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_scene[str(record['scene'])].append(index)
    generator = random.Random(int(seed))
    scenes = sorted(by_scene)
    generator.shuffle(scenes)
    for indices in by_scene.values():
        generator.shuffle(indices)
    order = []
    maximum = max((len(indices) for indices in by_scene.values()), default=0)
    for slot in range(maximum):
        round_scenes = scenes.copy()
        generator.shuffle(round_scenes)
        for scene in round_scenes:
            indices = by_scene[scene]
            if slot < len(indices):
                order.append(indices[slot])
    if len(order) != len(records) or len(set(order)) != len(records):
        raise RuntimeError('scene-balanced order must contain every record exactly once')
    return order
