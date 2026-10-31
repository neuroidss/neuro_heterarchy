#!/usr/bin/env python3
"""
[DEPRECATED] neuro_genesis_engine.py
Все классы разделены по отдельным модулям:
- UniversalSemanticSpace -> diffusion_semantic_space.py
- AmodalMontyColumnGPU -> neuro_monty_columns.py
- SelfSufficientHeterarchyRouter -> neuro_amodal_engine.py
- PHYSICAL_PROJECTION_768x120 -> neuro_amodal_engine.py
"""

from diffusion_semantic_space import UniversalSemanticSpace
from neuro_amodal_engine import SelfSufficientHeterarchyRouter, PHYSICAL_PROJECTION_768x120
