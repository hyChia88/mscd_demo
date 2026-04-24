import ifcopenshell
import ifcopenshell.util.element
import os
from typing import Optional, Dict, List, Any


class IFCEngine:
    """
    IFC Data Gateway - Unified interface for IFC model access and graph operations.

    Supports:
    - Local IFC file parsing and spatial indexing
    - Neo4j graph database export for semantic reasoning
    - Property extraction for compliance checking (SGPset, Pset_*)

    Architecture:
        IFCEngine (Data Gateway)
            ├── Local Spatial Index (in-memory)
            ├── Neo4j Graph Export (optional)
            └── Property Extraction Layer
    """

    def __init__(self, file_path: str, neo4j_conn=None, llm_client=None):
        """
        Initialize IFC Engine with optional Neo4j connection.

        Args:
            file_path: Path to IFC file
            neo4j_conn: Optional py2neo Graph connection for graph export
            llm_client: Optional LLM client for registry parsing.
                Must expose .complete(prompt: str) -> str.
                If None, registry parsing falls back to regex.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"IFC file not found: {file_path}")

        print(f"🏗️  Loading IFC Model: {os.path.basename(file_path)}...")
        self.file = ifcopenshell.open(file_path)
        self.file_path = file_path
        # Derive model key from filename for multi-model Neo4j isolation.
        # e.g. "AdvancedProject.ifc" → "AP", "BH_Office.ifc" → "BH"
        self.model_key = self._derive_model_key(file_path)
        self.spatial_index = {}
        self.neo4j_conn = neo4j_conn
        self._llm_client = llm_client  # set before _build_spatial_graph so registries use it
        self._size_cluster_taxonomy = self._load_size_cluster_taxonomy()
        self._build_spatial_graph()

    @staticmethod
    def _load_size_cluster_taxonomy():
        """Load `mscd_demo/prompts/size_cluster_taxonomy.json` if present.

        Returns the raw dict. G9 retrieval relies on each IfcWindow / IfcDoor
        node carrying a `size_cluster` label that matches the LoRA's emitted
        label vocabulary, so reload-once-at-init is the right shape.
        """
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "prompts", "size_cluster_taxonomy.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "size_cluster_taxonomy.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        import json as _json
                        return _json.load(f)
                except Exception:
                    pass
        return None

    def _snap_size_cluster(self, ifc_class: str, width_mm: Optional[float],
                           height_mm: Optional[float]) -> Optional[str]:
        """Return the nearest taxonomy cluster label, or None if unavailable."""
        if not self._size_cluster_taxonomy or width_mm is None or height_mm is None:
            return None
        clusters = (self._size_cluster_taxonomy.get("clusters") or {}).get(ifc_class) or {}
        if not clusters:
            return None
        best_label: Optional[str] = None
        best_d2 = float("inf")
        wx = float(width_mm)
        hx = float(height_mm)
        for label, info in clusters.items():
            dw = float(info.get("w", 0.0)) - wx
            dh = float(info.get("h", 0.0)) - hx
            d2 = dw * dw + dh * dh
            if d2 < best_d2:
                best_d2 = d2
                best_label = label
        return best_label

    @staticmethod
    def _derive_model_key(file_path: str) -> str:
        """Derive a short model key from IFC filename for Neo4j node tagging.

        Known models: AdvancedProject → AP, BH* → BH, DXA* → DXA.
        Falls back to uppercase first two chars of the stem.
        """
        stem = os.path.splitext(os.path.basename(file_path))[0].upper()
        if "ADVANCEDPROJECT" in stem or stem.startswith("AP"):
            return "AP"
        if "BASICHOUSE" in stem or stem.startswith("BH"):
            return "BH"
        if "DUPLEX" in stem or stem.startswith("DXA"):
            return "DXA"
        return stem[:2]

    def _build_spatial_graph(self):
        """
        构建空间拓扑索引 (The Topological Semantics Layer).

        Strategy:
        1. Always index by IfcBuildingStorey (primary spatial structure)
        2. Use IfcRelContainedInSpatialStructure for accurate element-storey mapping
        3. Also index IfcSpace if available (rooms within storeys)
        """
        print("⚙️  Building Semantic Graph Index...")

        # Primary: Index by IfcRelContainedInSpatialStructure (most accurate)
        # This captures the actual spatial containment relationships
        for rel in self.file.by_type("IfcRelContainedInSpatialStructure"):
            structure = rel.RelatingStructure
            structure_name = structure.Name if structure.Name else "Unknown"
            key = structure_name.lower()

            if key not in self.spatial_index:
                self.spatial_index[key] = []

            for element in rel.RelatedElements:
                if element.is_a("IfcOpeningElement"):
                    continue

                self.spatial_index[key].append({
                    "guid": element.GlobalId,
                    "type": element.is_a(),
                    "name": element.Name if element.Name else "Unnamed",
                    "description": element.Description if hasattr(element, "Description") else ""
                })

        # Secondary: Also index IfcSpace (rooms) if available
        spaces = self.file.by_type("IfcSpace")
        if spaces:
            for space in spaces:
                room_name = space.LongName if space.LongName else space.Name
                if not room_name:
                    continue

                key = room_name.lower()
                if key not in self.spatial_index:
                    self.spatial_index[key] = []

                # Get elements within this space via decomposition
                elements = ifcopenshell.util.element.get_decomposition(space)
                for el in elements:
                    if el.is_a("IfcOpeningElement") or el.is_a("IfcSpace"):
                        continue

                    # Avoid duplicates
                    existing_guids = {e["guid"] for e in self.spatial_index[key]}
                    if el.GlobalId not in existing_guids:
                        self.spatial_index[key].append({
                            "guid": el.GlobalId,
                            "type": el.is_a(),
                            "name": el.Name if el.Name else "Unnamed",
                            "description": el.Description if hasattr(el, "Description") else ""
                        })

        # Fallback: If no spatial structure found, group by element type
        if not self.spatial_index:
            element_types = {
                "walls": self.file.by_type("IfcWall") + self.file.by_type("IfcWallStandardCase"),
                "slabs": self.file.by_type("IfcSlab"),
                "doors": self.file.by_type("IfcDoor"),
                "windows": self.file.by_type("IfcWindow"),
            }

            for group_name, elements in element_types.items():
                if elements:
                    key = group_name.lower()
                    self.spatial_index[key] = []

                    for el in elements:
                        self.spatial_index[key].append({
                            "guid": el.GlobalId,
                            "type": el.is_a(),
                            "name": el.Name if el.Name else "Unnamed",
                            "description": el.Description if hasattr(el, "Description") else ""
                        })

        self._build_storey_registry()   # LLM parse: floor_num + category
        import time; time.sleep(1)      # avoid rate-limit between two LLM registry calls
        self._build_space_registry()    # LLM parse: space_type + unit_id
        print(f"✅  Graph Index Ready: {len(self.spatial_index)} groups indexed.")

    # =========================================================================
    # Phase 0 — Storey & Space Registry (LLM Structured Parsing)
    # =========================================================================

    def _llm_complete(self, prompt: str) -> str:
        """
        LLM completion stub for registry parsing.

        Wire up your LLM client here, or set self._llm_client before IFC load:
            engine = IFCEngine("model.ifc")
            engine._llm_client = my_client   # must expose .complete(prompt) -> str

        If not configured, raises NotImplementedError and the caller's
        regex fallback takes over automatically.
        """
        if hasattr(self, '_llm_client') and self._llm_client:
            return self._llm_client.complete(prompt)
        raise NotImplementedError(
            "LLM client not configured — regex fallback will be used."
        )

    def _build_storey_registry(self):
        """
        Parse every IfcBuildingStorey name into structured attributes via one LLM batch call.
        Falls back to regex patterns if LLM is unavailable.

        Stores:
            self.storey_registry  : {canonical → {"floor_num": int|None, "category": str, "raw": str}}
            self._storey_by_num   : {floor_num (int) → canonical}
            self._storey_by_cat   : {category (str)  → [canonicals]}
        """
        import json
        import re
        import yaml
        from pathlib import Path

        storeys = list(self.file.by_type("IfcBuildingStorey"))
        names = [s.Name or f"Storey_{s.id()}" for s in storeys]

        # ── Load prompt from YAML & call LLM ──────────────────────────────────
        parsed = []
        _llm_error: str | None = None
        try:
            _yaml = yaml.safe_load(
                (Path(__file__).parent.parent / "prompts" / "ifc_registry.yaml").read_text()
            )
            prompt = _yaml["storey_registry_prompt"].format(names=json.dumps(names))
            raw = self._llm_complete(prompt)
            parsed = json.loads(raw)
        except Exception as exc:
            _llm_error = str(exc)

        # ── Regex fallback (if LLM unavailable / parse error) ─────────────────
        _CAT_RE = [
            (re.compile(r'ground|g/?f|lobby|entrance',          re.I), 'ground'),
            (re.compile(r'basement|b\d|underground|carpark|park|garage', re.I), 'basement'),
            (re.compile(r'roof|rooftop|sky\s*lobby',            re.I), 'roof'),
            (re.compile(r'mezzanine|mezz',                      re.I), 'mezzanine'),
            (re.compile(r'podium',                              re.I), 'podium'),
        ]

        def _fallback_parse(name: str) -> dict:
            m = re.search(r'(-?\d+)', name)
            floor_num = int(m.group(1)) if m else None
            for pat, cat in _CAT_RE:
                if pat.search(name):
                    return {"floor_num": floor_num, "category": cat}
            return {"floor_num": floor_num, "category": "normal"}

        if len(parsed) != len(names):
            print(f"⚠️  storey_registry: LLM parse failed ({_llm_error}), "
                  f"using regex fallback for {len(names)} storeys.")
            parsed = [_fallback_parse(n) for n in names]

        # ── Build registry & secondary indices ────────────────────────────────
        self.storey_registry = {}
        self._storey_by_num  = {}   # {floor_num: [canonical, ...]} — 1:many
        self._storey_by_cat  = {}

        for name, info in zip(names, parsed):
            canonical = name.lower()
            floor_num = info.get("floor_num")
            category  = info.get("category", "normal")
            self.storey_registry[canonical] = {
                "raw": name, "floor_num": floor_num, "category": category
            }
            if floor_num is not None:
                self._storey_by_num.setdefault(floor_num, []).append(canonical)
            self._storey_by_cat.setdefault(category, []).append(canonical)

    def _resolve_storey_query(self, query: str):
        """
        Resolve a natural-language storey reference → list of canonical siblings.

        IFC files may have multiple storeys sharing the same floor_num
        (e.g. "Level 1" and "1 - First Floor" both at floor_num=1).
        This method returns ALL siblings so Cypher can match any of them.

        Strategy (ordered by cost):
        1. Exact canonical match    O(1)   "6 - sixth floor" → siblings of floor_num
        2. Floor number extraction  O(1)   "level 6", "6f" → floor_num=6 → siblings
        3. Category keyword         O(k)   "ground floor" → category lookup
        4. difflib fuzzy            O(n)   typo fallback, last resort

        Returns list[str] of canonical keys (siblings), or [query] if unresolved.
        """
        import re
        from difflib import get_close_matches

        q = query.lower().strip()
        if not q:
            return []
        registry = getattr(self, 'storey_registry', {})
        if not registry:
            return [q]

        def _siblings_of(canonical: str) -> list:
            """Return all canonicals sharing the same floor_num."""
            info = registry.get(canonical, {})
            fnum = info.get("floor_num")
            if fnum is not None:
                return list(self._storey_by_num.get(fnum, [canonical]))
            return [canonical]

        # 1. Exact
        if q in registry:
            return _siblings_of(q)

        # 2. Floor number
        m = re.search(r'(-?\d+)', q)
        if m:
            num = int(m.group(1))
            siblings = self._storey_by_num.get(num, [])
            if siblings:
                return list(siblings)

        # 3. Category keyword (universal English vocabulary, not project-specific)
        _QUERY_CAT = {
            "ground": "ground", "g/f": "ground", "gf": "ground", "grade": "ground",
            "basement": "basement", "carpark": "basement", "underground": "basement", "garage": "basement",
            "roof": "roof", "rooftop": "roof",
            "mezzanine": "mezzanine", "mezz": "mezzanine",
            "podium": "podium",
        }
        by_cat = getattr(self, '_storey_by_cat', {})
        for kw, cat in _QUERY_CAT.items():
            if kw in q:
                candidates = by_cat.get(cat, [])
                if candidates:
                    if m and len(candidates) > 1:
                        num = int(m.group(1))
                        for c in candidates:
                            if registry[c].get("floor_num") == num:
                                return _siblings_of(c)
                    return list(candidates)
                break

        # 4. difflib fallback
        close = get_close_matches(q, list(registry.keys()), n=1, cutoff=0.60)
        if close:
            return _siblings_of(close[0])
        return [q]

    def _build_space_registry(self):
        """
        Parse every IfcSpace name into structured attributes via one LLM batch call.
        Uses LongName if available (more descriptive), falls back to Name.
        Falls back to regex patterns if LLM is unavailable.

        Stores:
            self.space_registry  : {canonical → {"space_type": str, "unit_id": str|None, "raw": str}}
            self._space_by_type  : {space_type → [canonicals]}
            self._space_by_unit  : {unit_id    → [canonicals]}
        """
        import json
        import re
        import yaml
        from pathlib import Path

        spaces = list(self.file.by_type("IfcSpace"))
        if not spaces:
            self.space_registry = {}
            self._space_by_type = {}
            self._space_by_unit = {}
            return

        names = [s.LongName or s.Name or f"Space_{s.id()}" for s in spaces]

        # ── Load prompt from YAML & call LLM ──────────────────────────────────
        parsed = []
        _llm_error: str | None = None
        try:
            _yaml = yaml.safe_load(
                (Path(__file__).parent.parent / "prompts" / "ifc_registry.yaml").read_text()
            )
            space_types_str = " | ".join(_yaml["space_types"])
            prompt = _yaml["space_registry_prompt"].format(
                space_types=space_types_str,
                names=json.dumps(names),
            )
            raw = self._llm_complete(prompt)
            parsed = json.loads(raw)
        except Exception as exc:
            _llm_error = str(exc)

        # ── Regex fallback ─────────────────────────────────────────────────────
        _TYPE_RE = [
            (re.compile(r'master.?bed|主卧|master\s*room',    re.I), 'master_bedroom'),
            (re.compile(r'bed|bedroom|卧室|睡房',              re.I), 'bedroom'),
            (re.compile(r'bath|wc|toilet|lavatory|卫生间|厕', re.I), 'bathroom'),
            (re.compile(r'kitchen|kitch|厨房',                re.I), 'kitchen'),
            (re.compile(r'living|lounge|客厅',                re.I), 'living_room'),
            (re.compile(r'dining|餐厅',                       re.I), 'dining_room'),
            (re.compile(r'corridor|hallway|走廊|过道',         re.I), 'corridor'),
            (re.compile(r'lobby|reception|大堂',              re.I), 'lobby'),
            (re.compile(r'office|study|书房|办公',             re.I), 'office'),
            (re.compile(r'store|storage|storeroom|储藏',      re.I), 'storage'),
            (re.compile(r'mech|plant|machine|机房',           re.I), 'mechanical'),
            (re.compile(r'park|carpark|garage|车库',          re.I), 'parking'),
            (re.compile(r'stair|楼梯',                        re.I), 'stairwell'),
            (re.compile(r'lift|elevator|elev|电梯',           re.I), 'elevator'),
            (re.compile(r'balcony|terrace|阳台',              re.I), 'balcony'),
        ]
        _UNIT_RE = re.compile(r'\b([A-Z]?\d{1,4}[A-Z]?)\b')

        def _fallback_parse_space(name: str) -> dict:
            for pat, stype in _TYPE_RE:
                if pat.search(name):
                    um = _UNIT_RE.search(name)
                    return {"space_type": stype, "unit_id": um.group(1) if um else None}
            um = _UNIT_RE.search(name)
            return {"space_type": "unknown", "unit_id": um.group(1) if um else None}

        if len(parsed) != len(names):
            print(f"⚠️  space_registry: LLM parse failed ({_llm_error}), "
                  f"using regex fallback for {len(names)} spaces.")
            parsed = [_fallback_parse_space(n) for n in names]

        # ── Build registry & secondary indices ────────────────────────────────
        self.space_registry = {}
        self._space_by_type = {}
        self._space_by_unit = {}

        for name, info in zip(names, parsed):
            canonical  = name.lower()
            space_type = info.get("space_type", "unknown")
            unit_id    = (info.get("unit_id") or "").lower() or None
            self.space_registry[canonical] = {
                "raw": name, "space_type": space_type, "unit_id": unit_id
            }
            self._space_by_type.setdefault(space_type, []).append(canonical)
            if unit_id:
                self._space_by_unit.setdefault(unit_id, []).append(canonical)

    def _resolve_space_query(self, query: str) -> list:
        """
        Resolve a room/space reference → list of matching canonical keys in space_registry.

        Returns a list because multiple spaces can share the same type
        (e.g. "the kitchen" may match kitchens on multiple floors).

        Strategy:
        1. Exact canonical match        → [exact_key]
        2. Space type keyword detection → _space_by_type[type]
        3. Unit ID detection            → _space_by_unit[unit_id]
        4. difflib fuzzy                → [best_matches]
        """
        import re
        from difflib import get_close_matches

        q = query.lower().strip()
        registry = getattr(self, 'space_registry', {})
        if not registry:
            return []

        # 1. Exact
        if q in registry:
            return [q]

        # 2. Space type keyword (longest match first)
        _QUERY_TYPE = {
            "master bedroom": "master_bedroom", "master bed": "master_bedroom",
            "bedroom": "bedroom", "bed room": "bedroom",
            "bathroom": "bathroom", "bath": "bathroom",
            "toilet": "toilet", "wc": "toilet",
            "kitchen": "kitchen",
            "living room": "living_room", "living": "living_room", "lounge": "living_room",
            "dining room": "dining_room", "dining": "dining_room",
            "corridor": "corridor", "hallway": "corridor",
            "lobby": "lobby",
            "office": "office",
            "storage": "storage", "store room": "storage", "storeroom": "storage",
            "mechanical": "mechanical", "plant room": "mechanical",
            "parking": "parking", "carpark": "parking",
            "stairwell": "stairwell", "staircase": "stairwell",
            "elevator": "elevator", "lift": "elevator",
            "balcony": "balcony", "terrace": "balcony",
        }
        by_type = getattr(self, '_space_by_type', {})
        for kw in sorted(_QUERY_TYPE, key=len, reverse=True):
            if kw in q:
                candidates = by_type.get(_QUERY_TYPE[kw], [])
                if candidates:
                    return candidates

        # 3. Unit ID
        um = re.search(r'\b([a-z]?\d{1,4}[a-z]?)\b', q)
        if um:
            uid = um.group(1)
            by_unit = getattr(self, '_space_by_unit', {})
            if uid in by_unit:
                return by_unit[uid]

        # 4. difflib fallback
        close = get_close_matches(q, list(registry.keys()), n=3, cutoff=0.60)
        return close

    def find_elements_in_space(self, room_query: str):
        """
        Spatial lookup: storey name or room/space name.

        Resolution order:
        1. Exact substring on spatial_index  (fast path, original behaviour)
        2. Structured storey resolution      "level 6" → floor_num lookup
        3. Structured space resolution       "the kitchen" → space_type lookup
        4. difflib on spatial_index keys     typo/fuzzy fallback
        """
        from difflib import get_close_matches

        q = room_query.lower().strip()
        if not q:
            return []

        # 1. Exact substring
        matches = [elems for key, elems in self.spatial_index.items() if q in key]
        if matches:
            return [e for elems in matches for e in elems]

        # 2. Structured storey resolution (returns list of siblings)
        storey_keys = self._resolve_storey_query(q)
        storey_matches = [e for k in storey_keys if k in self.spatial_index
                          for e in self.spatial_index[k]]
        if storey_matches:
            return storey_matches

        # 3. Structured space resolution (may return multiple spaces of same type)
        space_keys = self._resolve_space_query(q)
        if space_keys:
            results = [e for k in space_keys if k in self.spatial_index
                       for e in self.spatial_index[k]]
            if results:
                return results

        # 4. difflib fallback on all spatial_index keys
        close = get_close_matches(q, list(self.spatial_index.keys()), n=1, cutoff=0.60)
        if close:
            return self.spatial_index[close[0]]

        return []

    def get_element_properties(self, guid: str) -> Dict[str, Any]:
        """
        获取元素完整属性集，包括 Pset_* 和 SGPset_* 属性。

        Args:
            guid: Element GlobalId

        Returns:
            Dict containing element properties and property sets
        """
        try:
            element = self.file.by_guid(guid)
            if not element:
                return {"error": f"Element not found: {guid}"}

            # 基本属性
            props = {
                "GlobalId": element.GlobalId,
                "Name": element.Name,
                "Type": element.is_a(),
                "ObjectType": getattr(element, "ObjectType", None),
                "Description": getattr(element, "Description", None),
            }

            # 提取所有 Property Sets
            psets = self._extract_property_sets(element)
            if psets:
                props["PropertySets"] = psets

            return props

        except Exception as e:
            return {"error": f"Error retrieving element: {str(e)}"}

    def _extract_property_sets(self, element) -> Dict[str, Dict]:
        """
        提取元素的所有属性集 (Pset_*, SGPset_*, etc.)

        Based on Zhu et al. (2023) IFC-Graph approach for semantic property extraction.
        """
        psets = {}

        # 方法1: 通过 IfcRelDefinesByProperties 获取属性
        if hasattr(element, "IsDefinedBy"):
            for definition in element.IsDefinedBy:
                if definition.is_a("IfcRelDefinesByProperties"):
                    prop_def = definition.RelatingPropertyDefinition

                    if prop_def.is_a("IfcPropertySet"):
                        pset_name = prop_def.Name
                        psets[pset_name] = {}

                        for prop in prop_def.HasProperties:
                            if prop.is_a("IfcPropertySingleValue"):
                                value = prop.NominalValue.wrappedValue if prop.NominalValue else None
                                psets[pset_name][prop.Name] = value

        return psets

    def get_element_by_guid(self, guid: str):
        """
        通过 GUID 获取元素对象

        Args:
            guid: Element GlobalId

        Returns:
            IFC element object or None
        """
        try:
            return self.file.by_guid(guid)
        except Exception:
            return None

    # =========================================================================
    # Neo4j Graph Export Methods
    # =========================================================================

    def export_to_neo4j(self, clear_existing: bool = False) -> Dict[str, int]:
        """
        将 IFC 语义模型导出到 Neo4j 图数据库。

        Based on Zhu et al. (2023) IFC-Graph methodology:
        - Nodes: IfcProduct subtypes (Wall, Door, Window, Space, etc.)
        - Relationships: Spatial containment, aggregation, connections
        - Properties: Pset_*, SGPset_* compliance properties

        Args:
            clear_existing: If True, clear existing nodes before import

        Returns:
            Dict with counts of created nodes and relationships
        """
        if not self.neo4j_conn:
            print("❌ Neo4j connection not configured")
            return {"error": "No Neo4j connection"}

        print("🏗️  Exporting IFC semantic model to Neo4j...")

        stats = {"nodes": 0, "relationships": 0}

        if clear_existing:
            self._clear_neo4j_graph()

        # 1. Create spatial structure nodes (Site -> Building -> Storey -> Space)
        stats["nodes"] += self._create_spatial_nodes()

        # 2. Create building element nodes
        stats["nodes"] += self._create_element_nodes()

        # 3. Create relationships
        stats["relationships"] += self._create_spatial_relationships()
        stats["relationships"] += self._create_element_relationships()

        print(f"✅ Neo4j export complete: {stats['nodes']} nodes, {stats['relationships']} relationships")
        return stats

    def _clear_neo4j_graph(self):
        """Clear all IFC-related nodes from Neo4j"""
        if self.neo4j_conn:
            self.neo4j_conn.run("MATCH (n:IFCElement) DETACH DELETE n")
            self.neo4j_conn.run("MATCH (n:IFCSpace) DETACH DELETE n")
            self.neo4j_conn.run("MATCH (n:IFCStorey) DETACH DELETE n")
            print("   Cleared existing IFC nodes")

    def _create_spatial_nodes(self) -> int:
        """Create nodes for spatial structure (Site, Building, Storey, Space)"""
        count = 0

        # Building Storeys
        for storey in self.file.by_type("IfcBuildingStorey"):
            self._create_node("IFCStorey", {
                "guid": storey.GlobalId,
                "name": storey.Name,
                "elevation": getattr(storey, "Elevation", 0),
                "ifc_model": self.model_key,
            })
            count += 1

        # Spaces
        for space in self.file.by_type("IfcSpace"):
            self._create_node("IFCSpace", {
                "guid": space.GlobalId,
                "name": space.Name or space.LongName,
                "long_name": space.LongName,
                "ifc_model": self.model_key,
            })
            count += 1

        return count

    def _create_element_nodes(self) -> int:
        """Create nodes for building elements with properties"""
        count = 0
        # Common element types across IFC2X3 and IFC4
        element_types = [
            "IfcWall", "IfcWallStandardCase",
            "IfcDoor", "IfcWindow",
            "IfcSlab", "IfcColumn", "IfcBeam",
            "IfcRailing", "IfcStair",  # required for ADJACENT_TO predicates
            "IfcFurnishingElement",  # IFC2X3 compatible
            "IfcFurniture",  # IFC4 only
        ]

        # Build element_guid -> storey_name mapping for the storey property.
        # Primary: ifcopenshell.util.element.get_container() which traverses
        # IfcRelAggregates as well as IfcRelContainedInSpatialStructure.
        # Fallback: elevation-based assignment for minimal IFC files (BH).
        import ifcopenshell.util.element as _ifc_util
        storey_map: Dict[str, str] = {}
        for ifc_type_pre in element_types:
            try:
                for element in self.file.by_type(ifc_type_pre):
                    container = _ifc_util.get_container(element)
                    if container and container.is_a("IfcBuildingStorey"):
                        storey_map[element.GlobalId] = container.Name or "Unknown"
            except RuntimeError:
                continue

        # Fallback: if get_container found nothing, assign by elevation
        if not storey_map:
            storeys = self.file.by_type("IfcBuildingStorey")
            if storeys:
                storey_elev = sorted(
                    [(s.Name, getattr(s, "Elevation", 0) or 0) for s in storeys],
                    key=lambda x: x[1],
                )
                for ifc_type_pre in element_types:
                    try:
                        for element in self.file.by_type(ifc_type_pre):
                            z = self._get_element_z(element)
                            if z is None:
                                continue
                            best_name = storey_elev[0][0]
                            for sn, se in storey_elev:
                                if se <= z + 100:
                                    best_name = sn
                                else:
                                    break
                            storey_map[element.GlobalId] = best_name
                    except RuntimeError:
                        continue
                if storey_map:
                    print(f"   ⚠️  storey_map built by elevation fallback "
                          f"({len(storey_map)} elements)")

        for ifc_type in element_types:
            try:
                elements = self.file.by_type(ifc_type)
            except RuntimeError:
                # Type not found in this IFC schema version, skip
                continue

            for element in elements:
                # Extract properties
                psets = self._extract_property_sets(element)

                node_props = {
                    "guid": element.GlobalId,
                    "name": element.Name,
                    "ifc_type": element.is_a(),
                    "object_type": getattr(element, "ObjectType", None),
                    "storey": storey_map.get(element.GlobalId),
                    "ifc_model": self.model_key,
                }

                # T1.3: Extract material via IfcRelAssociatesMaterial
                material_name = self._get_element_material(element)
                if material_name:
                    node_props["material"] = material_name

                # Fix 3: physical dimensions for IfcWindow/IfcDoor (breaks size-entropy)
                # G9: also snap to size_cluster taxonomy for classification-based retrieval.
                if element.is_a() in ("IfcWindow", "IfcDoor"):
                    w, h = self._get_element_dimensions(element, psets)
                    if w is not None:
                        node_props["width_mm"] = w
                    if h is not None:
                        node_props["height_mm"] = h
                    cluster = self._snap_size_cluster(element.is_a(), w, h)
                    if cluster:
                        node_props["size_cluster"] = cluster

                # Flatten key properties for graph queries
                for pset_name, props in psets.items():
                    for prop_name, prop_value in props.items():
                        # Key compliance properties (IFC-SG schema)
                        if prop_name in ["FireRating", "LoadBearing", "AcousticRating",
                                         "ThermalTransmittance", "IsExternal"]:
                            node_props[prop_name] = prop_value

                self._create_node("IFCElement", node_props)
                count += 1

        return count

    def _get_element_material(self, element) -> Optional[str]:
        """
        T1.3 — Extract material name(s) from an IFC element via IfcRelAssociatesMaterial.

        Handles multiple IFC material representations:
        - IfcMaterialLayerSetUsage / IfcMaterialLayerSet → join layer names
        - IfcMaterialList → join material names
        - IfcMaterial → single name

        Returns a pipe-separated string of material names, or None.
        """
        try:
            if not hasattr(element, 'HasAssociations'):
                return None
            for assoc in element.HasAssociations:
                if not assoc.is_a('IfcRelAssociatesMaterial'):
                    continue
                mat = assoc.RelatingMaterial
                if mat.is_a('IfcMaterialLayerSetUsage'):
                    mat = mat.ForLayerSet
                if mat.is_a('IfcMaterialLayerSet'):
                    names = [layer.Material.Name for layer in mat.MaterialLayers
                             if layer.Material and layer.Material.Name]
                    return '|'.join(names) if names else None
                elif mat.is_a('IfcMaterialList'):
                    names = [m.Name for m in mat.Materials if m.Name]
                    return '|'.join(names) if names else None
                elif mat.is_a('IfcMaterial'):
                    return mat.Name
        except Exception:
            pass
        return None

    def _get_element_dimensions(self, element, psets: dict) -> tuple:
        """
        Fix 3 — Extract physical width and height (mm) from IfcWindow or IfcDoor.

        Primary source: OverallWidth / OverallHeight attributes (IFC standard).
        Fallback: scan all psets for OverallWidth / OverallHeight / Width / Height.

        Returns (width_mm, height_mm) — either or both may be None.
        IFC stores dimensions in mm for AdvancedProject (verified: units=MILLIMETRE).
        """
        width: Optional[float] = None
        height: Optional[float] = None
        try:
            # IFC direct attribute (most reliable)
            w_attr = getattr(element, "OverallWidth", None)
            h_attr = getattr(element, "OverallHeight", None)
            if w_attr is not None:
                width = float(w_attr)
            if h_attr is not None:
                height = float(h_attr)
        except Exception:
            pass
        # Fallback: pset scan (handles IFC2x3 files without direct attribute)
        if width is None or height is None:
            dim_keys_w = {"OverallWidth", "Width", "Breite"}
            dim_keys_h = {"OverallHeight", "Height", "Höhe"}
            for pset_props in psets.values():
                for k, v in pset_props.items():
                    try:
                        if width is None and k in dim_keys_w:
                            width = float(v)
                        if height is None and k in dim_keys_h:
                            height = float(v)
                    except (TypeError, ValueError):
                        pass
        return width, height

    def _create_spatial_relationships(self) -> int:
        """Create CONTAINS relationships between spaces/storeys and elements.

        Primary path: IfcRelContainedInSpatialStructure (standard IFC).
        Fallback: per-element elevation-based assignment for any element that
        has an IFCElement node but did NOT get a CONTAINS edge from the
        primary path (handles partial containment coverage).
        """
        count = 0
        contained_guids: set = set()

        type_counts: dict = {}  # diagnostic: per-type CONTAINS edge counts
        for rel in self.file.by_type("IfcRelContainedInSpatialStructure"):
            space = rel.RelatingStructure
            space_guid = space.GlobalId

            for element in rel.RelatedElements:
                self._create_relationship(
                    "IFCSpace" if space.is_a("IfcSpace") else "IFCStorey",
                    space_guid,
                    "CONTAINS",
                    "IFCElement",
                    element.GlobalId
                )
                contained_guids.add(element.GlobalId)
                t = element.is_a()
                type_counts[t] = type_counts.get(t, 0) + 1
                count += 1

        if type_counts:
            summary = ", ".join(f"{t}={n}" for t, n in sorted(type_counts.items()))
            print(f"   CONTAINS primary: {count} edges ({summary})")

        # Fallback: elevation-based storey assignment for orphaned elements
        # (elements that have IFCElement nodes but no CONTAINS edge).
        if self.neo4j_conn:
            fallback_count = self._assign_elements_to_storeys_by_elevation(
                exclude_guids=contained_guids
            )
            if fallback_count > 0:
                print(f"   ⚠️  Elevation fallback assigned {fallback_count} orphaned "
                      f"elements (primary path covered {count})")
            count += fallback_count

        return count

    def _assign_elements_to_storeys_by_elevation(
        self, exclude_guids: set = None
    ) -> int:
        """Assign elements to storeys by comparing element Z vs storey elevations.

        For each element whose GlobalId is NOT in *exclude_guids*, pick the
        storey whose elevation is ≤ element Z and closest (i.e. the floor the
        element sits on).  Also sets the element node's ``storey`` property so
        it matches the canonical name.

        Args:
            exclude_guids: GUIDs that already have CONTAINS edges (skip them).
        """
        storeys = self.file.by_type("IfcBuildingStorey")
        if not storeys:
            return 0

        exclude = exclude_guids or set()

        # Sort storeys ascending by elevation
        storey_list = sorted(
            [(s.GlobalId, s.Name, getattr(s, "Elevation", 0) or 0) for s in storeys],
            key=lambda x: x[2],
        )

        element_types = [
            "IfcWall", "IfcWallStandardCase", "IfcDoor", "IfcWindow",
            "IfcSlab", "IfcColumn", "IfcBeam", "IfcRailing", "IfcStair",
            "IfcFurnishingElement", "IfcFurniture",
        ]

        count = 0
        for ifc_type in element_types:
            try:
                elements = self.file.by_type(ifc_type)
            except RuntimeError:
                continue
            for element in elements:
                if element.GlobalId in exclude:
                    continue
                z = self._get_element_z(element)
                # Find the storey with highest elevation ≤ element Z.
                # If Z is unavailable (e.g. BH doors/windows with no
                # ObjectPlacement), fall back to the lowest storey.
                if z is None:
                    best_storey = storey_list[0]  # lowest storey
                else:
                    best_storey = storey_list[0]  # default to lowest
                    for sg, sn, se in storey_list:
                        if se <= z + 100:  # 100mm tolerance
                            best_storey = (sg, sn, se)
                        else:
                            break
                sg, sn, se = best_storey
                self._create_relationship(
                    "IFCStorey", sg, "CONTAINS", "IFCElement", element.GlobalId
                )
                # Also update the element node's storey property
                if self.neo4j_conn:
                    self.neo4j_conn.run(
                        "MATCH (e:IFCElement {guid: $guid}) SET e.storey = $storey",
                        guid=element.GlobalId, storey=sn,
                    )
                count += 1

        return count

    @staticmethod
    def _get_element_z(element) -> Optional[float]:
        """Extract the Z coordinate from an element's ObjectPlacement."""
        try:
            placement = element.ObjectPlacement
            if placement and placement.is_a("IfcLocalPlacement"):
                rp = placement.RelativePlacement
                if rp and rp.is_a("IfcAxis2Placement3D"):
                    loc = rp.Location
                    if loc:
                        coords = loc.Coordinates
                        return float(coords[2]) if len(coords) >= 3 else None
        except (AttributeError, IndexError, TypeError):
            pass
        return None

    def _create_element_relationships(self) -> int:
        """Create relationships between elements (voids, fills, connections, NEXT_TO)"""
        count = 0

        # Build opening_guid -> host_wall_guid mapping first.
        # IfcOpeningElement nodes are NOT created in Neo4j (they are intermediate
        # geometry objects, not building elements), so we skip them and create
        # direct Door/Window -[:FILLS]-> Wall edges instead.
        opening_to_host: Dict[str, str] = {}
        for rel in self.file.by_type("IfcRelVoidsElement"):
            host = rel.RelatingBuildingElement
            opening = rel.RelatedOpeningElement
            if host and opening:
                opening_to_host[opening.GlobalId] = host.GlobalId

        # IfcRelFillsElement: Door/Window -[:FILLS]-> Wall
        # Chain: Door/Window -[FillsElement]-> IfcOpeningElement -[VoidsElement]-> Wall
        # We compress this to a direct edge, skipping the opening element node.
        from collections import defaultdict
        wall_fillers: Dict[str, List] = defaultdict(list)  # wall_guid -> [filler_elements]

        for rel in self.file.by_type("IfcRelFillsElement"):
            filler = rel.RelatedBuildingElement    # IfcDoor or IfcWindow
            opening = rel.RelatingOpeningElement   # IfcOpeningElement (intermediate)
            if filler and opening:
                host_guid = opening_to_host.get(opening.GlobalId)
                if host_guid:
                    self._create_relationship(
                        "IFCElement", filler.GlobalId,
                        "FILLS",
                        "IFCElement", host_guid
                    )
                    count += 1
                    wall_fillers[host_guid].append(filler)

        # T10.1 + T10.2: NEXT_TO edges between consecutive fillers on the same wall
        count += self._create_next_to_edges(wall_fillers)

        # T10.6: wall_child_count property on wall nodes
        self._set_wall_child_counts(wall_fillers)

        # CONNECTS_TO: wall-to-wall path connections from IfcRelConnectsPathElements
        count += self._create_connects_to_edges()

        # Fix 4: ADJACENT_TO edges between same-storey elements within 1500mm
        count += self._create_adjacent_to_edges()

        return count

    def _create_connects_to_edges(self) -> int:
        """Create CONNECTS_TO edges between walls from IfcRelConnectsPathElements.

        These are wall-to-wall structural connections (T-junctions, L-corners, etc.)
        exported by Revit/ArchiCAD. 686 edges in AdvancedProject.ifc.

        Connection types (stored as edge property):
          ATSTART/ATEND — wall A's start connects to wall B's end
          ATPATH — wall A connects at some point along wall B's length (T-junction)

        Edges are deduplicated: same wall pair gets one edge with connection_type.
        """
        count = 0
        seen_pairs: set = set()

        try:
            rels = self.file.by_type("IfcRelConnectsPathElements")
        except RuntimeError:
            return 0

        for rel in rels:
            a = rel.RelatingElement
            b = rel.RelatedElement
            if not a or not b:
                continue

            # Skip self-connections
            if a.GlobalId == b.GlobalId:
                continue

            # Deduplicate: only one edge per unique pair
            pair_key = tuple(sorted([a.GlobalId, b.GlobalId]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            conn_type_a = getattr(rel, "RelatingConnectionType", None)
            conn_type_b = getattr(rel, "RelatedConnectionType", None)
            conn_str = f"{conn_type_a}/{conn_type_b}" if conn_type_a else ""

            # Bidirectional CONNECTS_TO
            props = {"connection_type": conn_str} if conn_str else {}
            self._create_relationship_with_props(
                "IFCElement", a.GlobalId,
                "CONNECTS_TO",
                "IFCElement", b.GlobalId,
                props,
            )
            self._create_relationship_with_props(
                "IFCElement", b.GlobalId,
                "CONNECTS_TO",
                "IFCElement", a.GlobalId,
                props,
            )
            count += 2

        return count

    def _create_adjacent_to_edges(self) -> int:
        """Fix 4: Create ADJACENT_TO edges between same-storey elements with centroid
        distance in (100mm, 1500mm].

        Mirrors the ADJACENT_TO criterion used in 2_hunt_skeletons.py so that
        Neo4j Cypher queries can now filter on this predicate.

        Includes distance_mm as an edge property for future distance-range filtering.
        Only links elements whose types are both in a relevant set (walls, windows,
        doors, railings, stairs) — avoids O(N²) over all 1200+ elements.

        Returns count of ADJACENT_TO edges created (bidirectional).
        """
        import ifcopenshell.util.placement
        import math
        from collections import defaultdict

        # Element types to include in ADJACENT_TO computation
        adj_types = {
            "IfcWall", "IfcWallStandardCase",
            "IfcDoor", "IfcWindow",
            "IfcRailing", "IfcStair", "IfcColumn",
        }

        # Collect (guid, storey, centroid) for all relevant elements in this model
        element_data: List[Dict] = []

        # Build storey_map using ifcopenshell.util.element.get_container() which
        # traverses IfcSpace → IfcBuildingStorey hierarchy (handles DXA / IFC2x3)
        import ifcopenshell.util.element as _ifc_elem_util
        primary_storey: Dict[str, str] = {}
        for rel in self.file.by_type("IfcRelContainedInSpatialStructure"):
            for el in rel.RelatedElements:
                if el.GlobalId in primary_storey:
                    continue
                container = _ifc_elem_util.get_container(el)
                # Walk up until we hit a BuildingStorey
                while container and not container.is_a("IfcBuildingStorey"):
                    container = _ifc_elem_util.get_container(container)
                if container and container.is_a("IfcBuildingStorey"):
                    primary_storey[el.GlobalId] = container.Name or ""

        for ifc_type in adj_types:
            try:
                elements = self.file.by_type(ifc_type)
            except RuntimeError:
                continue
            for element in elements:
                storey = primary_storey.get(element.GlobalId)
                if not storey:
                    continue  # skip unplaced elements
                try:
                    mat = ifcopenshell.util.placement.get_local_placement(
                        element.ObjectPlacement
                    )
                    cx, cy, cz = mat[0][3], mat[1][3], mat[2][3]
                except Exception:
                    continue
                element_data.append({
                    "guid": element.GlobalId,
                    "ifc_type": element.is_a(),
                    "storey": storey,
                    "cx": cx,
                    "cy": cy,
                    "cz": cz,
                })

        # Group by storey for O(per-storey N²) instead of O(total N²)
        by_storey: Dict[str, List[Dict]] = defaultdict(list)
        for ed in element_data:
            by_storey[ed["storey"]].append(ed)

        DIST_MIN = 100.0   # mm — exclude degenerate same-origin pairs
        DIST_MAX = 1500.0  # mm — same threshold as skeleton miner

        count = 0
        seen_pairs: set = set()

        for storey_name, group in by_storey.items():
            n = len(group)
            for i in range(n):
                a = group[i]
                for j in range(i + 1, n):
                    b = group[j]
                    dx = a["cx"] - b["cx"]
                    dy = a["cy"] - b["cy"]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= DIST_MIN or dist > DIST_MAX:
                        continue
                    pair_key = tuple(sorted([a["guid"], b["guid"]]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    props = {"distance_mm": round(dist, 1)}
                    self._create_relationship_with_props(
                        "IFCElement", a["guid"],
                        "ADJACENT_TO",
                        "IFCElement", b["guid"],
                        props,
                    )
                    self._create_relationship_with_props(
                        "IFCElement", b["guid"],
                        "ADJACENT_TO",
                        "IFCElement", a["guid"],
                        props,
                    )
                    count += 2

        print(f"   ADJACENT_TO: {count // 2} pairs → {count} edges "
              f"(across {len(by_storey)} storeys)")
        return count

    def _create_next_to_edges(self, wall_fillers: Dict[str, List]) -> int:
        """T10.1: Create NEXT_TO edges between consecutive door/window fillers on the same wall.

        Algorithm:
        1. For each wall with 2+ fillers, group fillers by storey
           (multi-story walls have vertically stacked windows — NOT neighbors)
        2. Within each storey group, project centroids onto wall's X-axis
        3. Sort by projected coordinate
        4. Create bidirectional NEXT_TO edges between consecutive pairs
        5. Set position_index property (0-based from one end)

        Returns count of NEXT_TO edges created.
        """
        import ifcopenshell.util.placement
        import ifcopenshell.util.element as _ifc_util
        import numpy as np
        from collections import defaultdict

        count = 0
        for wall_guid, fillers in wall_fillers.items():
            if len(fillers) < 2:
                continue

            # Get wall direction (local X-axis from placement matrix)
            try:
                wall = self.file.by_guid(wall_guid)
                wall_mat = ifcopenshell.util.placement.get_local_placement(
                    wall.ObjectPlacement
                )
                wall_dir = np.array([wall_mat[0][0], wall_mat[1][0], wall_mat[2][0]])
                wall_origin = np.array([wall_mat[0][3], wall_mat[1][3], wall_mat[2][3]])
            except Exception:
                continue

            # Group fillers by storey (multi-story walls span floors)
            storey_groups: Dict[str, List] = defaultdict(list)
            for filler in fillers:
                container = _ifc_util.get_container(filler)
                storey_key = container.Name if container else "_unknown"
                storey_groups[storey_key].append(filler)

            # Create NEXT_TO within each storey group
            for storey_name, group in storey_groups.items():
                if len(group) < 2:
                    continue

                # Project onto wall axis
                filler_projections = []
                for filler in group:
                    try:
                        mat = ifcopenshell.util.placement.get_local_placement(
                            filler.ObjectPlacement
                        )
                        centroid = np.array([mat[0][3], mat[1][3], mat[2][3]])
                        proj = np.dot(centroid - wall_origin, wall_dir)
                        filler_projections.append((proj, filler))
                    except Exception:
                        continue

                if len(filler_projections) < 2:
                    continue

                filler_projections.sort(key=lambda x: x[0])

                # Create NEXT_TO edges between consecutive pairs
                for i in range(len(filler_projections) - 1):
                    _, filler_a = filler_projections[i]
                    _, filler_b = filler_projections[i + 1]

                    # Bidirectional NEXT_TO
                    self._create_relationship_with_props(
                        "IFCElement", filler_a.GlobalId,
                        "NEXT_TO",
                        "IFCElement", filler_b.GlobalId,
                        {"position_index": i, "wall_guid": wall_guid},
                    )
                    self._create_relationship_with_props(
                        "IFCElement", filler_b.GlobalId,
                        "NEXT_TO",
                        "IFCElement", filler_a.GlobalId,
                        {"position_index": i, "wall_guid": wall_guid},
                    )
                    count += 2

                # Set position_index on each filler node
                for idx, (_, filler) in enumerate(filler_projections):
                    if self.neo4j_conn:
                        self.neo4j_conn.run(
                            "MATCH (e:IFCElement {guid: $guid}) "
                            "SET e.wall_position_index = $idx, "
                            "    e.wall_child_total = $total",
                            guid=filler.GlobalId,
                            idx=idx,
                            total=len(filler_projections),
                        )

        return count

    def _set_wall_child_counts(self, wall_fillers: Dict[str, List]):
        """T10.6: Set wall_child_count property on wall nodes."""
        if not self.neo4j_conn:
            return
        for wall_guid, fillers in wall_fillers.items():
            self.neo4j_conn.run(
                "MATCH (w:IFCElement {guid: $guid}) SET w.wall_child_count = $count",
                guid=wall_guid, count=len(fillers),
            )

    def _create_node(self, label: str, properties: Dict):
        """Create a Neo4j node with given label and properties"""
        if not self.neo4j_conn:
            return

        # Filter out None values
        props = {k: v for k, v in properties.items() if v is not None}

        query = f"""
        MERGE (n:{label} {{guid: $guid}})
        SET n += $props
        """
        self.neo4j_conn.run(query, guid=props.get("guid"), props=props)

    def _create_relationship(self, from_label: str, from_guid: str,
                             rel_type: str, to_label: str, to_guid: str):
        """Create a Neo4j relationship between two nodes"""
        if not self.neo4j_conn:
            return

        query = f"""
        MATCH (a:{from_label} {{guid: $from_guid}})
        MATCH (b:{to_label} {{guid: $to_guid}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        self.neo4j_conn.run(query, from_guid=from_guid, to_guid=to_guid)

    def _create_relationship_with_props(self, from_label: str, from_guid: str,
                                        rel_type: str, to_label: str, to_guid: str,
                                        props: Dict[str, Any]):
        """Create a Neo4j relationship with properties"""
        if not self.neo4j_conn:
            return

        query = f"""
        MATCH (a:{from_label} {{guid: $from_guid}})
        MATCH (b:{to_label} {{guid: $to_guid}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        """
        self.neo4j_conn.run(query, from_guid=from_guid, to_guid=to_guid, props=props)

    # =========================================================================
    # Graph Query Methods (for Agent Reasoning)
    # =========================================================================

    def query_elements_by_level(self, level_name: str) -> List[Dict]:
        """
        Query elements on a specific level/storey via Neo4j.

        Useful for RQ3 abductive reasoning: "Which elements are on Level 6?"
        """
        if not self.neo4j_conn:
            return self.find_elements_in_space(level_name)

        # Structured resolve before Cypher — handles "level 6" → ["6 - sixth floor", "level 6"]
        canonicals = self._resolve_storey_query(level_name.lower()) or [level_name]
        query = """
        MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
        WHERE ANY(c IN $level_names WHERE toLower(s.name) CONTAINS c)
          AND e.ifc_model = $model
        RETURN e.guid as guid, e.name as name, e.ifc_type as type,
               e.FireRating as fire_rating, e.LoadBearing as load_bearing
        """
        result = self.neo4j_conn.run(query, level_names=canonicals, model=self.model_key)
        return [dict(record) for record in result]

    def query_elements_by_property(self, property_name: str, property_value: Any) -> List[Dict]:
        """
        Query elements with specific property value.

        Useful for RQ2 compliance checking: "Find all doors with FireRating < 60min"
        """
        if not self.neo4j_conn:
            return []

        query = f"""
        MATCH (e:IFCElement)
        WHERE e.{property_name} = $value
          AND e.ifc_model = $model
        RETURN e.guid as guid, e.name as name, e.ifc_type as type
        """
        result = self.neo4j_conn.run(query, value=property_value, model=self.model_key)
        return [dict(record) for record in result]

    def query_adjacent_elements(self, guid: str) -> List[Dict]:
        """
        Find elements adjacent to or connected with given element.

        Useful for spatial context in defect localization.
        """
        if not self.neo4j_conn:
            return []

        query = """
        MATCH (e:IFCElement {guid: $guid})-[:HAS_OPENING|FILLS|CONTAINS*1..2]-(related:IFCElement)
        WHERE related.ifc_model = $model
        RETURN DISTINCT related.guid as guid, related.name as name, related.ifc_type as type
        """
        result = self.neo4j_conn.run(query, guid=guid, model=self.model_key)
        return [dict(record) for record in result]

    # =========================================================================
    # Phase 1a — New Query Primitives (required by Phase 5 RetrievalBackend)
    # =========================================================================

    def query_elements_in_space(self, space_name: str, ifc_type: str = "") -> List[Dict]:
        """
        Query elements within a named room/space.

        Memory mode: delegates to find_elements_in_space() (Phase 0 registry-aware).
        Neo4j mode:  Cypher MATCH on IFCSpace-[:CONTAINS]->IFCElement.

        Args:
            space_name: Room/space query string (e.g. "living room", "kitchen")
            ifc_type:   Optional IFC class filter (e.g. "IfcWindow"). Empty = no filter.
        """
        if not self.neo4j_conn:
            results = self.find_elements_in_space(space_name)
            if ifc_type:
                results = [r for r in results if r.get("type") == ifc_type]
            return results

        type_clause = "AND (e.ifc_type = $ifc_type OR e.ifc_type STARTS WITH $ifc_type)" if ifc_type else ""
        query = f"""
        MATCH (sp:IFCSpace)-[:CONTAINS]->(e:IFCElement)
        WHERE toLower(sp.name) CONTAINS toLower($space_name)
          AND e.ifc_model = $model
        {type_clause}
        RETURN e.guid as guid, e.name as name, e.ifc_type as type,
               sp.name as space
        """
        result = self.neo4j_conn.run(query, space_name=space_name.lower(),
                                     ifc_type=ifc_type, model=self.model_key)
        return [dict(r) for r in result]

    def query_elements_by_name_keyword(self, keyword: str) -> List[Dict]:
        """
        Search elements whose name contains a keyword (fuzzy, deduped).

        Memory mode: scans all elements in spatial_index by .name field.
        Neo4j mode:  Cypher CONTAINS on e.name, LIMIT 20.

        Args:
            keyword: Equipment brand, ID, or name fragment (e.g. "Daikin", "AHU-03")
        """
        if not self.neo4j_conn:
            kw = keyword.lower()
            results: List[Dict] = []
            seen: set = set()
            for elems in self.spatial_index.values():
                for e in elems:
                    if e["guid"] not in seen and kw in (e.get("name") or "").lower():
                        results.append(e)
                        seen.add(e["guid"])
            return results

        query = """
        MATCH (e:IFCElement)
        WHERE toLower(e.name) CONTAINS toLower($keyword)
          AND e.ifc_model = $model
        RETURN e.guid as guid, e.name as name, e.ifc_type as type
        LIMIT 20
        """
        result = self.neo4j_conn.run(query, keyword=keyword, model=self.model_key)
        return [dict(r) for r in result]

    def query_elements_by_neighbor(self, ifc_type: str, neighbor_type: str) -> List[Dict]:
        """
        Find elements of ifc_type structurally connected to neighbor_type (Neo4j only).

        Uses HAS_OPENING and FILLS — the only element↔element adjacency edges
        currently in the graph (covers door/window in opening relationships).
        NEAR edges for full spatial adjacency are deferred to Step 3.

        Memory fallback: returns [] — topology requires the graph.

        Args:
            ifc_type:      Target element class (e.g. "IfcWindow")
            neighbor_type: Adjacent element class (e.g. "IfcColumn", "IfcDoor")
        """
        if not self.neo4j_conn:
            return []  # No memory fallback — topology requires graph

        query = """
        MATCH (e:IFCElement)-[:HAS_OPENING|FILLS]-(nb:IFCElement)
        WHERE (e.ifc_type = $type OR e.ifc_type STARTS WITH $type)
          AND nb.ifc_type = $neighbor_type
          AND e.ifc_model = $model
        RETURN DISTINCT e.guid as guid, e.name as name, e.ifc_type as type
        """
        result = self.neo4j_conn.run(query, type=ifc_type,
                                     neighbor_type=neighbor_type,
                                     model=self.model_key)
        return [dict(r) for r in result]