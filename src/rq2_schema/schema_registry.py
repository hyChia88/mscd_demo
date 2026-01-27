"""
Schema Registry for loading and managing JSON Schemas.
"""

import json
from pathlib import Path
from typing import Any, Dict


class SchemaRegistry:
    """
    Loads and manages JSON Schema for validation.
    """

    def __init__(self, schema_path: str):
        """
        Initialize schema registry with a schema file.

        Args:
            schema_path: Path to the JSON Schema file

        Raises:
            FileNotFoundError: If schema file doesn't exist
        """
        self.path = Path(schema_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Schema not found: {self.path}")

        self.schema: Dict[str, Any] = json.loads(
            self.path.read_text(encoding="utf-8")
        )
        self.schema_id: str = self.schema.get("$id", self.path.as_posix())

    def get_required_fields(self) -> list:
        """Get list of required top-level fields from schema."""
        return self.schema.get("required", [])
