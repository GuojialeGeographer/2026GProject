"""PyRestore — task-configurable multimodal street-view indicator framework.

Turns a georeferenced image manifest (or a validated before/after pair table) into an
analysis-ready, provenance-recorded, GIS-exportable dataset by coordinating two evidence
families:

* fixed-taxonomy CV physical measurements (segmentation shares, GVI/SVF/enclosure,
  colour/edge statistics), and
* task-configurable VLM semantic interpretations (schema-shaped structured outputs).

Tasks are declarative YAML definitions (``tasks/*.yaml``); adding a task does not change
framework code. See ``docs/ARCHITECTURE.md`` for the design contract.
"""

__version__ = "0.2.1"

from .config import load_config
from .manifest import load_manifest
from .pairs import load_pairs, make_pairs
from .pipeline import run_task
from .vlm import load_task

__all__ = [
    "load_config",
    "load_manifest",
    "load_pairs",
    "load_task",
    "make_pairs",
    "run_task",
]
