import numpy as np

from myria3d.pctl.dataset.flair3d_label_remap import apply_remap, get_definition


def test_segment_lut_merges_buildings_and_maps_void():
    definition = get_definition("segment", "default")
    raw = np.array([0, 14, 15, 16, -1, 300], dtype=np.int64)
    remapped = apply_remap(raw, definition)
    np.testing.assert_array_equal(remapped, [0, 14, 0, 0, 15, 15])
