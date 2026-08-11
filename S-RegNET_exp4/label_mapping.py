"""
SynthSeg / FreeSurfer integer-label -> 5-class remapping (background, cortex,
subcortical GM, white matter, CSF). NumPy only, so the converter and inference
share one definition without heavy deps.
"""

import numpy as np

# FreeSurfer/SynthSeg label -> 5-class (0 bg, 1 cortex, 2 subcortical GM,
# 3 white matter, 4 CSF). Unlisted labels fall through to background.
LABEL_MAPPING = {
    0: 0, 24: 0,
    3: 1, 42: 1,
    10: 2, 49: 2, 11: 2, 50: 2, 12: 2, 51: 2, 13: 2, 52: 2,
    17: 2, 53: 2, 18: 2, 54: 2, 26: 2, 58: 2, 28: 2, 60: 2, 8: 2, 47: 2,
    2: 3, 41: 3, 7: 3, 46: 3, 16: 3,
    4: 4, 43: 4, 5: 4, 44: 4, 14: 4, 15: 4,
}


def remap_to_5class(int_volume):
    """Map an integer-label volume to 0..4 (unlisted -> background). Returns a
    new array, same shape and dtype as int_volume."""
    remapped = np.zeros_like(int_volume)
    for src, dst in LABEL_MAPPING.items():
        remapped[int_volume == src] = dst
    return remapped
