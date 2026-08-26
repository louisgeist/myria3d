"""
Flair3D+ unified label definitions and LUT remapping. (LUT: look-up table)

Registry for `segment` (per-point semantic classes) and `natural_habitat` (raw CarHab
ids, remapped into 4 low-cardinality ecological axes consumed as tile_distribution
targets — see myria3d/pctl/dataset/pointcept_npy.py).

Source-agnostic: remapping applies the same whether labels come from a PLY field
or a sampled GeoTIFF raster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Class names (default / fine-grained taxonomies)
# ---------------------------------------------------------------------------

_SEGMENT_NAMES: Tuple[str, ...] = (
    "Building",
    "Greenhouse",
    "Impervious surface",
    "Other soil",
    "Herbaceous",
    "Vineyard",
    "Other vegetation",
    "Other infrastructures",
    "Swimming pool",
    "Water",
    "Deciduous",
    "Coniferous",
    "Bridge",
    "Agricultural soil",
    "Soil under vegetation",
    "Void",
)

# Source taxonomy uses 15=Industrial and 16=Tertiary; both merge into Building (0).
# Void is not a source id (npy uses negatives). LUT index 255 is a safe missing-fill
# raw id that already maps to train void (15).
_SEGMENT_VOID_TRAIN_ID = 15
_SEGMENT_MISSING_FILL_RAW_ID = 255

# Raw CarHab natural_habitat ids (0-43) are structured as: ids 0-35 = 6 domain/type
# blocks (open-temperate, forest-temperate, open-mediterranean, forest-mediterranean,
# open-alpine, forest-alpine) of 6 ids each (acid+humid, acid+mesic, acid+dry,
# alkaline+humid, alkaline+mesic, alkaline+dry); ids 36-39 = mineral/aquatic (acid,
# alkaline each); id 40 = cultivated; id 41 = built/artificial; id 42 = N/A; id 43 =
# roads & paved tracks. Ids 42 and 43 are void in every axis below.

_NATURAL_HABITAT_BY_HABITAT_TYPE_ECOLOGICAL_NAMES: Tuple[str, ...] = (
    "Open habitat",
    "Forest habitat",
    "Mineral habitat",
    "Aquatic habitat",
    "Void",
)

_NATURAL_HABITAT_BY_MOISTURE_REGIME_NAMES: Tuple[str, ...] = (
    "Humid",
    "Mesic",
    "Dry",
    "Void",
)

_NATURAL_HABITAT_BY_SOIL_CHEMISTRY_NAMES: Tuple[str, ...] = (
    "Acidic",
    "Alkaline",
    "Void",
)

_NATURAL_HABITAT_BY_CLIMATIC_DOMAIN_NAMES: Tuple[str, ...] = (
    "Temperate domain",
    "Mediterranean domain",
    "Alpine domain",
    "Void",
)


@dataclass(frozen=True)
class LabelDefinition:
  """Named label remap: raw source ids -> consecutive train ids via LUT."""

  name: str
  task_key: str
  num_raw_classes: int
  lut: np.ndarray
  names: Tuple[str, ...]
  ignore_index: int
  missing_fill_raw_id: int
  source_field: str = ""

  def __post_init__(self) -> None:
    if self.lut.shape != (self.num_raw_classes,):
      raise ValueError(
        f"LUT shape {self.lut.shape} != ({self.num_raw_classes},) "
        f"for {self.task_key}/{self.name}"
      )
    if len(self.names) == 0:
      raise ValueError(f"names must be non-empty for {self.task_key}/{self.name}")


@dataclass(frozen=True)
class PreprocessLabelDefinitions:
  """Bundle of label definitions passed to preprocessing workers."""

  segment: LabelDefinition

  def to_meta_dict(self) -> Dict[str, str]:
    return {"segment": self.segment.name}


def build_lut_from_groups(
    num_raw: int,
    groups: Mapping[int, Sequence[int]],
    *,
    default_train_id: int,
) -> np.ndarray:
  """Build a LUT mapping raw ids to train ids via explicit groups."""
  lut = np.full(num_raw, default_train_id, dtype=np.int32)
  for train_id, raw_ids in groups.items():
    for raw_id in raw_ids:
      if raw_id < 0 or raw_id >= num_raw:
        raise ValueError(
          f"raw_id {raw_id} out of range [0, {num_raw}) for train_id {train_id}"
        )
      lut[raw_id] = int(train_id)
  return lut


def _build_segment_lut(num_raw: int = 256, void_train_id: int = 15) -> np.ndarray:
  """Identity for 0..14; industrial/tertiary (15/16) -> Building (0); rest -> void."""
  lut = np.full(num_raw, void_train_id, dtype=np.int32)
  lut[:void_train_id] = np.arange(void_train_id, dtype=np.int32)
  lut[15] = 0
  lut[16] = 0
  return lut


def _make_definition(
    task_key: str,
    name: str,
    *,
    num_raw_classes: int,
    lut: np.ndarray,
    names: Tuple[str, ...],
    ignore_index: int,
    missing_fill_raw_id: int,
    source_field: str = "",
) -> LabelDefinition:
  return LabelDefinition(
    name=name,
    task_key=task_key,
    num_raw_classes=num_raw_classes,
    lut=lut.astype(np.int32, copy=False),
    names=names,
    ignore_index=ignore_index,
    missing_fill_raw_id=missing_fill_raw_id,
    source_field=source_field,
  )


def _register_segment_definitions() -> Dict[str, LabelDefinition]:
  segment_lut = _build_segment_lut()
  base = _make_definition(
    "segment",
    "default",
    num_raw_classes=segment_lut.shape[0],
    lut=segment_lut,
    names=_SEGMENT_NAMES,
    ignore_index=_SEGMENT_VOID_TRAIN_ID,
    missing_fill_raw_id=_SEGMENT_MISSING_FILL_RAW_ID,
    source_field="semantic",
  )
  # Upstream PLY may already use inter_finerall10 train ids; same LUT/metadata.
  inter_finerall10 = _make_definition(
    "segment",
    "inter_finerall10",
    num_raw_classes=segment_lut.shape[0],
    lut=segment_lut.copy(),
    names=_SEGMENT_NAMES,
    ignore_index=_SEGMENT_VOID_TRAIN_ID,
    missing_fill_raw_id=_SEGMENT_MISSING_FILL_RAW_ID,
    source_field="semantic",
  )
  return {"default": base, "inter_finerall10": inter_finerall10}


def _register_natural_habitat_definitions() -> Dict[str, LabelDefinition]:
  """4 low-cardinality ecological axes derived from raw (44-class) CarHab ids.

  Raw id layout: see module-level comment above the axis name tuples. Ids 42 (N/A) and
  43 (roads) fall outside every axis's groups and land on that axis's `default_train_id`
  (void) automatically.
  """
  open_ids = list(range(0, 6)) + list(range(12, 18)) + list(range(24, 30))
  forest_ids = list(range(6, 12)) + list(range(18, 24)) + list(range(30, 36))

  habitat_type_lut = build_lut_from_groups(
    44,
    {0: open_ids, 1: forest_ids, 2: [36, 37], 3: [38, 39]},
    default_train_id=4,
  )

  humid_ids = [block + offset for block in range(0, 36, 6) for offset in (0, 3)]
  mesic_ids = [block + offset for block in range(0, 36, 6) for offset in (1, 4)]
  dry_ids = [block + offset for block in range(0, 36, 6) for offset in (2, 5)]
  moisture_regime_lut = build_lut_from_groups(
    44,
    {0: humid_ids, 1: mesic_ids, 2: dry_ids},
    default_train_id=3,
  )

  acidic_ids = [block + offset for block in range(0, 36, 6) for offset in (0, 1, 2)] + [36, 38]
  alkaline_ids = [block + offset for block in range(0, 36, 6) for offset in (3, 4, 5)] + [37, 39]
  soil_chemistry_lut = build_lut_from_groups(
    44,
    {0: acidic_ids, 1: alkaline_ids},
    default_train_id=2,
  )

  climatic_domain_lut = build_lut_from_groups(
    44,
    {0: list(range(0, 12)), 1: list(range(12, 24)), 2: list(range(24, 36))},
    default_train_id=3,
  )

  # Missing-file fallback: raw id 43 (roads) is void in every axis above, matching
  # Pointcept's own missing-target sentinel for this field.
  missing_fill_raw_id = 43

  return {
    "by_habitat_type_ecological": _make_definition(
      "natural_habitat",
      "by_habitat_type_ecological",
      num_raw_classes=44,
      lut=habitat_type_lut,
      names=_NATURAL_HABITAT_BY_HABITAT_TYPE_ECOLOGICAL_NAMES,
      ignore_index=4,
      missing_fill_raw_id=missing_fill_raw_id,
      source_field="NATURAL_HABITAT",
    ),
    "by_moisture_regime": _make_definition(
      "natural_habitat",
      "by_moisture_regime",
      num_raw_classes=44,
      lut=moisture_regime_lut,
      names=_NATURAL_HABITAT_BY_MOISTURE_REGIME_NAMES,
      ignore_index=3,
      missing_fill_raw_id=missing_fill_raw_id,
      source_field="NATURAL_HABITAT",
    ),
    "by_soil_chemistry": _make_definition(
      "natural_habitat",
      "by_soil_chemistry",
      num_raw_classes=44,
      lut=soil_chemistry_lut,
      names=_NATURAL_HABITAT_BY_SOIL_CHEMISTRY_NAMES,
      ignore_index=2,
      missing_fill_raw_id=missing_fill_raw_id,
      source_field="NATURAL_HABITAT",
    ),
    "by_climatic_domain": _make_definition(
      "natural_habitat",
      "by_climatic_domain",
      num_raw_classes=44,
      lut=climatic_domain_lut,
      names=_NATURAL_HABITAT_BY_CLIMATIC_DOMAIN_NAMES,
      ignore_index=3,
      missing_fill_raw_id=missing_fill_raw_id,
      source_field="NATURAL_HABITAT",
    ),
  }


LABEL_DEFINITIONS: Dict[str, Dict[str, LabelDefinition]] = {
  "segment": _register_segment_definitions(),
  "natural_habitat": _register_natural_habitat_definitions(),
}

# Default definition per task (preprocess v2 CLI + training when not overridden).
DEFAULT_LABEL_DEFINITION_NAMES: Dict[str, str] = {
  "segment": "default",
}

# The 4 natural_habitat axis definitions applied together to derive myria3d's
# tile_distribution tasks (nathab_habitat_type, nathab_moisture_regime,
# nathab_soil_chemistry, nathab_bioclimatic_zone) — see pointcept_npy.py.
NATURAL_HABITAT_AXIS_DEFINITIONS: Dict[str, str] = {
  "nathab_habitat_type": "by_habitat_type_ecological",
  "nathab_moisture_regime": "by_moisture_regime",
  "nathab_soil_chemistry": "by_soil_chemistry",
  "nathab_bioclimatic_zone": "by_climatic_domain",
}


def get_default_definition_name(task_key: str) -> str:
  if task_key not in DEFAULT_LABEL_DEFINITION_NAMES:
    keys = ", ".join(sorted(DEFAULT_LABEL_DEFINITION_NAMES.keys()))
    raise KeyError(f"Unknown task_key '{task_key}'. Expected one of: {keys}")
  return DEFAULT_LABEL_DEFINITION_NAMES[task_key]


def supported_definitions(task_key: str) -> Tuple[str, ...]:
  if task_key not in LABEL_DEFINITIONS:
    keys = ", ".join(sorted(LABEL_DEFINITIONS.keys()))
    raise KeyError(f"Unknown task_key '{task_key}'. Expected one of: {keys}")
  return tuple(sorted(LABEL_DEFINITIONS[task_key].keys()))


def get_definition(task_key: str, name: str) -> LabelDefinition:
  if task_key not in LABEL_DEFINITIONS:
    keys = ", ".join(sorted(LABEL_DEFINITIONS.keys()))
    raise KeyError(f"Unknown task_key '{task_key}'. Expected one of: {keys}")
  task_defs = LABEL_DEFINITIONS[task_key]
  if name not in task_defs:
    supported = ", ".join(sorted(task_defs.keys()))
    raise KeyError(
      f"Unknown definition '{name}' for task '{task_key}'. Supported: {supported}"
    )
  return task_defs[name]


def build_preprocess_label_definitions(
    segment: str = DEFAULT_LABEL_DEFINITION_NAMES["segment"],
) -> PreprocessLabelDefinitions:
  return PreprocessLabelDefinitions(segment=get_definition("segment", segment))


def apply_remap(values: np.ndarray, definition: LabelDefinition) -> np.ndarray:
  """Remap raw label ids to train ids via LUT; out-of-range -> missing_fill mapping."""
  idx = values.astype(np.int64, copy=False)
  lut = definition.lut
  fallback = int(lut[definition.missing_fill_raw_id])
  result = np.full(idx.shape, fallback, dtype=np.int32)
  valid = (idx >= 0) & (idx < lut.shape[0])
  if np.any(valid):
    result[valid] = lut[idx[valid]]
  return result


