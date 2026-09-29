from myria3d.pctl.dataset.downstream.base import DownstreamNpyDataset


class OpenGFDataset(DownstreamNpyDataset):
    """OpenGF (ground filtering, binary Ground / Non-ground): no usable color channel
    in the Pointcept preprocessing (the LAS RGB fields hold a constant dummy value) --
    ``color_mask`` is set all-True so the frozen backbone's learned ``color_mask_value``
    fill-in is used instead of a naive zero (see ``GridProbeModel._fill_masked_features``).
    Intensity is real and kept.
    """

    def __init__(self, data_root: str, split_dir: str, **kwargs):
        kwargs.setdefault("has_color", False)
        kwargs.setdefault("has_strength", True)
        kwargs.setdefault("label_key", "segment")
        super().__init__(data_root, split_dir, **kwargs)
