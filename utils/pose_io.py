"""Pose IO helpers: import/export joint values as JSON."""
from __future__ import annotations
from typing import Dict
import json

__all__ = ['import_joint_values', 'export_joint_values']


def import_joint_values(file_path: str) -> Dict[str, float]:
    """Load a JSON file containing {'joint_values': {name: value}} and return the dict.
    Raises ValueError on invalid schema.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict) or 'joint_values' not in data:
        raise ValueError("缺少 'joint_values' 字段")
    joint_values = data['joint_values']
    if not isinstance(joint_values, dict):
        raise ValueError("'joint_values' 应为字典")
    return {k: float(v) for k, v in joint_values.items()}


def export_joint_values(file_path: str, joint_values: Dict[str, float]) -> None:
    """Write {'joint_values': {...}} to JSON file."""
    from datetime import datetime
    payload = {
        'joint_values': joint_values,
        'timestamp': datetime.now().isoformat()
    }
    with open(file_path, 'w') as f:
        json.dump(payload, f, indent=2)
