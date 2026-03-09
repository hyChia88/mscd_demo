"""
V2 Data Structures

Core Pydantic models for constraints extraction, query planning, and retrieval.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal, Optional, Dict, Any


class SpatialTriplet(BaseModel):
    """
    A single architectural spatial predicate extracted by the Neuro layer.

    Encodes the interface relationship between two IFC element types, e.g.:
        (IfcWindow) -[ADJACENT_TO]-> (IfcRailing)

    Used as input to the Priority 0 Cypher compiler in constraints_to_query.py.
    All predicates correspond to pre-computed Neo4j edges (offline geometry).
    """

    subject_type: str  # e.g. "IfcWindow"
    predicate: Literal[
        "FILLS",           # door/window occupies opening in wall (IfcRelFillsElement)
        "CONTINUOUS",      # element spans multiple storeys (Top constraint ≠ storey)
        "ADJACENT_TO",     # same-storey centroid distance < 1.5 m
        "ON_TOP_OF",       # Z_min(subject) > Z_max(object) + XY AABB overlap
        "PERPENDICULAR_TO",  # wall orientation vectors dot-product ≈ 0
        "PARALLEL_TO",       # wall orientation vectors dot-product ≈ 1
    ]
    object_type: str  # e.g. "IfcRailing"
    object_material: Optional[str] = None  # optional material filter on the ref element
    confidence: float = 0.0  # VLM extraction confidence [0, 1]


class Constraints(BaseModel):
    """
    Extracted constraints from chat + images + 4D context.

    Represents structured semantic understanding of the user's query
    in terms of spatial and semantic filters.
    """

    storey_name: Optional[str] = None  # e.g., "6 - Sixth Floor", "Level 1"
    ifc_class: Optional[str] = None  # e.g., "IfcWindow", "IfcWall", "IfcDoor"
    near_keywords: List[str] = Field(default_factory=list)  # e.g., ["north", "elevator"]
    relations: List[str] = Field(default_factory=list)  # e.g., ["adjacent_to", "in_room"]

    # ── NEW FIELDS (Phase 2 — Granularity Upgrade) ─────────────────────────
    space_name: Optional[str] = None
    # Specific room/space name, one level below storey.
    # e.g. "Living Room", "Master Bedroom", "Corridor 6A"
    # Source: LLM extraction from chat + floorplan.spatial_zone fallback (Phase 3)
    # Enables: space+type query (Priority 0) → ~5 candidates

    target_name_keyword: Optional[str] = None
    # Equipment brand, ID, or unique element name fragment.
    # e.g. "Daikin", "AHU-03", "Fire Pump FP-01"
    # NOT for generic type words ("window", "wall") — those go to ifc_class.
    # Enables: name_keyword query (Priority 1) → ~1-3 candidates

    neighbor_type: Optional[str] = None
    # IFC class of a nearby reference element for topological location.
    # e.g. "IfcColumn", "IfcStair", "IfcDoor"
    # Source: "next to the column", "beside the door", "near the staircase"
    # Enables: neighbor+type query (Priority 3, Neo4j only, HAS_OPENING/FILLS)
    # ───────────────────────────────────────────────────────────────────────

    # ── V2.5 NEW FIELD (Phase 5 — Neuro-Symbolic) ───────────────────────────
    spatial_relations: List[SpatialTriplet] = Field(default_factory=list)
    # Structured spatial predicates extracted by the Neuro layer (LoRA_3).
    # Each triplet encodes (subject_type, predicate, object_type) observed
    # in the site photo/relation-crop images.
    # Enables: spatial_triplet query (Priority 0) → ~1-3 candidates
    # Backward-compatible: empty list → Priority 0 skipped, existing 1-8 cascade used.
    # ────────────────────────────────────────────────────────────────────────

    # Diagnostics
    confidence: float = 0.0  # Confidence score (0.0-1.0)
    source: str = "unknown"  # "prompt" | "lora" | "prompt_failed"


class QueryPlan(BaseModel):
    """Deterministic query plan for retrieval."""

    priority: int  # 1 (highest) to 5 (fallback)
    strategy: str  # "storey+type", "storey_only", "type_only", "keyword", "fallback"
    params: Dict[str, Any]  # Parameters for execution
    expected_pool_size: Optional[int] = None


class RetrievalResult(BaseModel):
    """Result from retrieval backend execution."""

    candidates: List[Dict[str, Any]]
    pool_size: int
    query_plan_used: QueryPlan
    backend: str  # "memory", "neo4j", "memory+clip", "neo4j+clip"
    rerank_applied: bool = False
    # ── Fallback transparency (added v2.6) ───────────────────────────────────
    # strategy_actually_used: the strategy that produced the candidates.
    # Differs from query_plan_used.strategy when predicate relaxation fired.
    # e.g. query_plan_used.strategy="spatial_triplet" but candidates came from
    # "storey+type" → strategy_actually_used="storey+type", fallback_triggered=True
    strategy_actually_used: str = ""   # empty = same as query_plan_used.strategy
    fallback_triggered: bool = False   # True if predicate relaxation demoted to lower strategy


class ParsedImage(BaseModel):
    """Structured VLM description of a single image."""

    image_path: str
    image_type: str  # "site_photo" | "floorplan"

    # Semantic fields (populated by VLM)
    element_type: Optional[str] = None       # e.g., "window", "wall", "slab"
    ifc_class_hint: Optional[str] = None     # e.g., "IfcWindow"
    material: Optional[str] = None           # e.g., "concrete", "glass"
    defect_type: Optional[str] = None        # e.g., "crack", "water_damage"
    defect_severity: Optional[str] = None    # "minor" | "moderate" | "severe"
    location_cues: List[str] = Field(default_factory=list)  # e.g., ["north facade"]

    # Floorplan-specific fields (None for site photos)
    spatial_zone: Optional[str] = None       # e.g., "north wing", "room 602"
    storey_hint: Optional[str] = None        # e.g., "6 - Sixth Floor"
    marked_elements: List[str] = Field(default_factory=list)

    # Summary for prompt injection
    description: str = ""                    # Free-text 2-3 sentence summary
    keywords: List[str] = Field(default_factory=list)

    # Diagnostics
    confidence: float = 0.0
    parse_latency_ms: float = 0.0


class ImageParseResult(BaseModel):
    """All parsed images for a single case."""

    site_photos: List[ParsedImage] = Field(default_factory=list)
    floorplan: Optional[ParsedImage] = None

    @property
    def all_images(self) -> List[ParsedImage]:
        images = list(self.site_photos)
        if self.floorplan:
            images.append(self.floorplan)
        return images

    @property
    def combined_description(self) -> str:
        """Single text block for prompt injection."""
        parts = []
        for i, photo in enumerate(self.site_photos, 1):
            parts.append(f"[Site Photo {i}]: {photo.description}")
            if photo.element_type:
                parts.append(f"  Element: {photo.element_type} ({photo.ifc_class_hint or 'unknown'})")
            if photo.defect_type:
                parts.append(f"  Defect: {photo.defect_type} ({photo.defect_severity or 'unknown'})")
            if photo.location_cues:
                parts.append(f"  Location cues: {', '.join(photo.location_cues)}")
        if self.floorplan:
            fp = self.floorplan
            parts.append(f"[Floorplan]: {fp.description}")
            if fp.storey_hint:
                parts.append(f"  Storey: {fp.storey_hint}")
            if fp.spatial_zone:
                parts.append(f"  Zone: {fp.spatial_zone}")
            if fp.marked_elements:
                parts.append(f"  Marked elements: {', '.join(fp.marked_elements)}")
        return "\n".join(parts)

    @property
    def inferred_ifc_class(self) -> Optional[str]:
        """Best IFC class hint across all images."""
        for img in self.all_images:
            if img.ifc_class_hint:
                return img.ifc_class_hint
        return None

    @property
    def inferred_storey(self) -> Optional[str]:
        """Best storey hint, preferring floorplan."""
        if self.floorplan and self.floorplan.storey_hint:
            return self.floorplan.storey_hint
        for img in self.site_photos:
            if img.storey_hint:
                return img.storey_hint
        return None

    @property
    def all_location_cues(self) -> List[str]:
        cues = []
        for img in self.all_images:
            cues.extend(img.location_cues)
        return list(dict.fromkeys(cues))  # deduplicate, preserve order


class V2Trace(BaseModel):
    """V2-specific trace data (augments EvalTrace)."""

    constraints: Optional[Constraints] = None
    query_plans: List[QueryPlan] = Field(default_factory=list)
    retrieval_results: List[RetrievalResult] = Field(default_factory=list)

    # Diagnostics
    constraints_parse_success: bool = False
    constraints_parse_error: Optional[str] = None
    rerank_gain: Optional[float] = None  # For condition B2/C2 (rank improvement)

    # Image parsing diagnostics
    image_parse_result: Optional[ImageParseResult] = None
    image_parse_ms: float = 0.0

    # Timing
    constraints_extraction_ms: float = 0.0
    query_planning_ms: float = 0.0
    retrieval_ms: float = 0.0


class ConditionOverride(BaseModel):
    """Condition-specific overrides (from profiles.yaml conditions section)."""

    model_config = ConfigDict(populate_by_name=True)

    use_images: bool = True
    use_floorplan: bool = False
    chat_blur: bool = False
    four_d_metadata: bool = Field(True, alias="4d_metadata")
    four_d_enhanced: bool = Field(False, alias="4d_enhanced")
    force_clip: bool = False


class ProfileConfig(BaseModel):
    """Profile configuration from profiles.yaml."""

    pipeline: str  # "v1" or "v2"
    constraints_model: Optional[str] = None  # "prompt" or "lora" (v2 only)
    retrieval: str  # "memory" or "neo4j"
    use_clip: bool = False
    thin_agent: bool = False
    rq2_schema: bool = True
    bcf: bool = False
    description: str = ""
