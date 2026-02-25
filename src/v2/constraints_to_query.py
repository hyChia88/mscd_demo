"""
Template-Based Query Planner

Translates extracted constraints into deterministic, priority-ordered query plans.
No LLM generation - uses fixed templates for reproducibility.
"""

from typing import List, Dict, Any
from .types import Constraints, QueryPlan, SpatialTriplet


class QueryPlanner:
    """
    Deterministic query planning with priority-based fallbacks.

    Generates ordered list of query plans from constraints, trying
    most specific queries first and falling back to broader queries.
    """

    # Priority rules: ordered from most specific to most general
    PRIORITY_RULES = [
        # ── Phase 5 NEW rule (Priority 0 — Neuro-Symbolic spatial triplet) ───
        {
            "priority": 0,
            "strategy": "spatial_triplet",
            "requires": ["spatial_relations", "ifc_class"],
            "description": "Topological triplet from Neuro layer — breaks attribute entropy bottleneck (~1-3 candidates)",
            "template_memory": None,  # no memory fallback — topology requires Neo4j graph
            "template_cypher": """
                MATCH (target:IFCElement)-[:{predicate}]->(ref:IFCElement)
                WHERE target.ifc_type = $subject_type
                  AND ref.ifc_type = $object_type
                  AND toLower(ref.storey) CONTAINS toLower($storey)
                RETURN target.guid as guid, target.name as name, target.ifc_type as type,
                       ref.ifc_type as ref_type, ref.storey as ref_storey
            """
        },
        # ── Phase 4 NEW rules (priorities 1-3, finer granularity) ────────────
        {
            "priority": 1,
            "strategy": "space+type",
            "requires": ["space_name", "ifc_class"],
            "description": "Most specific: element type within a named room/space (~5 candidates)",
            "template_memory": "filter_by_space_and_type",
            "template_cypher": """
                MATCH (sp:IFCSpace)-[:CONTAINS]->(e:IFCElement)
                WHERE toLower(sp.name) CONTAINS toLower($space_name)
                  AND e.ifc_type = $type
                RETURN e.guid as guid, e.name as name, e.ifc_type as type,
                       sp.name as space
            """
        },
        {
            "priority": 2,
            "strategy": "name_keyword",
            "requires": ["target_name_keyword"],
            "description": "Equipment brand/ID fuzzy name match (~1-3 candidates)",
            "template_memory": "search_by_name_keyword",
            "template_cypher": """
                MATCH (e:IFCElement)
                WHERE toLower(e.name) CONTAINS toLower($name_keyword)
                RETURN e.guid as guid, e.name as name, e.ifc_type as type
                LIMIT 20
            """
        },
        {
            "priority": 3,
            "strategy": "neighbor+type",
            "requires": ["neighbor_type", "ifc_class"],
            "description": "Topological: element adjacent to known neighbor type — Neo4j only (~3-8 candidates)",
            "template_memory": "filter_by_neighbor_type",
            "template_cypher": """
                MATCH (e:IFCElement)-[:HAS_OPENING|FILLS]-(nb:IFCElement)
                WHERE e.ifc_type = $type
                  AND nb.ifc_type = $neighbor_type
                RETURN DISTINCT e.guid as guid, e.name as name, e.ifc_type as type
            """
        },
        # ── Original rules (renumbered 1-5 → 4-8) ───────────────────────────
        {
            "priority": 4,
            "strategy": "storey+type",
            "requires": ["storey_name", "ifc_class"],
            "description": "Both storey and IFC type known (~50 candidates)",
            "template_memory": "filter_by_storey_and_type",
            "template_cypher": """
                MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
                WHERE toLower(s.name) CONTAINS toLower($storey)
                  AND e.ifc_type = $type
                RETURN e.guid as guid, e.name as name, e.ifc_type as type,
                       s.name as storey
            """
        },
        {
            "priority": 5,
            "strategy": "storey_only",
            "requires": ["storey_name"],
            "description": "Narrow to storey/floor only (~200 candidates)",
            "template_memory": "filter_by_storey",
            "template_cypher": """
                MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
                WHERE toLower(s.name) CONTAINS toLower($storey)
                RETURN e.guid as guid, e.name as name, e.ifc_type as type,
                       s.name as storey
            """
        },
        {
            "priority": 6,
            "strategy": "type_only",
            "requires": ["ifc_class"],
            "description": "Filter by IFC type across all storeys (~150 candidates)",
            "template_memory": "filter_by_type",
            "template_cypher": """
                MATCH (e:IFCElement)
                WHERE e.ifc_type = $type
                RETURN e.guid as guid, e.name as name, e.ifc_type as type
            """
        },
        {
            "priority": 7,
            "strategy": "keyword",
            "requires": ["near_keywords"],
            "description": "Text search using spatial keywords (~100 candidates)",
            "template_memory": "search_by_keywords",
            "template_cypher": """
                MATCH (e:IFCElement)
                WHERE toLower(e.name) CONTAINS toLower($keyword)
                   OR toLower(e.description) CONTAINS toLower($keyword)
                RETURN e.guid as guid, e.name as name, e.ifc_type as type
            """
        },
        {
            "priority": 8,
            "strategy": "fallback",
            "requires": [],
            "description": "Return first 100 elements (escalation candidate)",
            "template_memory": "get_all_elements",
            "template_cypher": """
                MATCH (e:IFCElement)
                RETURN e.guid as guid, e.name as name, e.ifc_type as type
                LIMIT 100
            """
        }
    ]

    def plan(self, constraints: Constraints) -> List[QueryPlan]:
        """
        Generate ordered list of query plans from constraints.

        Args:
            constraints: Extracted constraints

        Returns:
            List of QueryPlans ordered by priority (highest to lowest)
        """
        plans = []

        # Check each rule in priority order
        for rule in self.PRIORITY_RULES:
            if self._constraints_satisfy_rule(constraints, rule):
                params = self._build_params(constraints, rule)
                expected_pool = self._estimate_pool_size(rule["strategy"], params)

                plans.append(QueryPlan(
                    priority=rule["priority"],
                    strategy=rule["strategy"],
                    params=params,
                    expected_pool_size=expected_pool
                ))

        # Always include fallback as last resort if not already present
        if not plans or plans[-1].strategy != "fallback":
            plans.append(QueryPlan(
                priority=8,
                strategy="fallback",
                params={},
                expected_pool_size=100
            ))

        return plans

    def _constraints_satisfy_rule(
        self,
        constraints: Constraints,
        rule: Dict[str, Any]
    ) -> bool:
        """
        Check if constraints have all required fields for this rule.

        Args:
            constraints: Extracted constraints
            rule: Priority rule dict

        Returns:
            True if all required fields are present and non-empty
        """
        required_fields = rule.get("requires", [])

        for field in required_fields:
            value = getattr(constraints, field, None)

            # Check if field is missing or empty
            if value is None:
                return False
            if isinstance(value, list) and len(value) == 0:
                return False
            if isinstance(value, str) and value.strip() == "":
                return False

        return True

    def _build_params(
        self,
        constraints: Constraints,
        rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build execution parameters from constraints based on rule requirements.

        Args:
            constraints: Extracted constraints
            rule: Priority rule dict

        Returns:
            Parameter dict for query execution
        """
        params = {}
        required_fields = rule.get("requires", [])

        if "storey_name" in required_fields:
            params["storey"] = constraints.storey_name

        if "ifc_class" in required_fields:
            params["type"] = constraints.ifc_class

        if "near_keywords" in required_fields and constraints.near_keywords:
            # Use first keyword for now (can be extended to multiple)
            params["keyword"] = constraints.near_keywords[0]
            params["keywords"] = constraints.near_keywords

        if "relations" in required_fields and constraints.relations:
            params["relations"] = constraints.relations

        # Phase 4 new field mappings
        if "space_name" in required_fields:
            params["space_name"] = constraints.space_name

        if "target_name_keyword" in required_fields:
            params["name_keyword"] = constraints.target_name_keyword

        if "neighbor_type" in required_fields:
            params["neighbor_type"] = constraints.neighbor_type
            params["type"] = constraints.ifc_class  # neighbor query also needs ifc_class

        # Phase 5: spatial_triplet — extract predicate data from the first triplet
        if "spatial_relations" in required_fields and constraints.spatial_relations:
            triplet: SpatialTriplet = constraints.spatial_relations[0]
            params["subject_type"] = triplet.subject_type
            params["predicate"] = triplet.predicate
            params["object_type"] = triplet.object_type
            params["spatial_relations"] = [t.model_dump() for t in constraints.spatial_relations]
            if triplet.object_material:
                params["object_material"] = triplet.object_material
            # storey is used in the Cypher WHERE clause; fall back to "" if absent
            if "storey" not in params:
                params["storey"] = constraints.storey_name or ""

        return params

    def _estimate_pool_size(self, strategy: str, params: Dict[str, Any]) -> int:
        """
        Estimate expected pool size for a query strategy.

        These are rough estimates based on typical IFC model statistics.

        Args:
            strategy: Query strategy name
            params: Query parameters

        Returns:
            Estimated pool size
        """
        # Rough estimates (order of magnitude)
        estimates = {
            "spatial_triplet": 3,   # Most specific: topological edge → ~1-3 matches
            "space+type":    5,     # Room + element type
            "name_keyword":  3,     # Equipment brand/ID match
            "neighbor+type": 8,     # Topological adjacency (Neo4j only, FILLS/HAS_OPENING)
            "storey+type":  50,     # Floor + element type
            "storey_only":  200,    # All elements on one floor
            "type_only":    150,    # All elements of one type
            "keyword":      100,    # Keyword search — variable
            "fallback":     100     # Capped at 100
        }

        return estimates.get(strategy, 100)

    @staticmethod
    def get_rule_description(priority: int) -> str:
        """
        Get human-readable description for a priority level.

        Args:
            priority: Priority number (1-5)

        Returns:
            Description string
        """
        for rule in QueryPlanner.PRIORITY_RULES:
            if rule["priority"] == priority:
                return rule.get("description", "")
        return "Unknown priority"
