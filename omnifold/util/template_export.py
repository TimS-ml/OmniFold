"""Data schema for serializing structural template hit information.

Provides the ``TemplateExport`` dataclass used to persist template hits
(PDB ID, chain, CIF path, residue mapping, E-value) to disk via pickle
for later injection into AF3 configuration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

@dataclass
class TemplateExport:
    """A light schema for serializing template hit data."""
    pdb_id: str            # '4pqx'
    chain_id: str          # 'A'
    cif_path: Path         # absolute path inside template_store/pdb
    query_idx_to_template_idx: Mapping[int,int]  # 0‑based
    e_value: float
    hit_from_chain: str    # OmniFold input chain id (e.g. 'A') 