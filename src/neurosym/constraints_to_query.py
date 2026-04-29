"""
Template-Based Query Planner

Translates extracted constraints into deterministic, priority-ordered query plans.
No LLM generation - uses fixed templates for reproducibility.
"""

import re
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
        # ── Priority 0a — edge-traversal predicates (FILLS, ADJACENT_TO, ON_TOP_OF, …) ──
        {
            "priority": 0,
            "strategy": "spatial_triplet",
            "requires": ["spatial_relations", "ifc_class"],
            # Fires for all predicates EXCEPT "CONTINUOUS" (stored as property, not edge)
            "predicate_exclude": ["CONTINUOUS"],
            "description": "Topological triplet from Neuro layer — breaks attribute entropy bottleneck (~1-3 candidates)",
            "template_memory": None,
            "template_cypher": """
                MATCH (target:IFCElement)-[:{predicate}]->(ref:IFCElement)
                WHERE target.ifc_type = $subject_type
                  AND ref.ifc_type = $object_type
                  AND target.ifc_model = $model
                  AND toLower(ref.storey) CONTAINS toLower($storey)
                RETURN target.guid as guid, target.name as name, target.ifc_type as type,
                       ref.ifc_type as ref_type, ref.storey as ref_storey
            """
        },
        # ── Priority 0b — CONTINUOUS: property-based (no edge, uses is_continuous flag) ──
        {
            "priority": 0,
            "strategy": "continuous_span",
            "requires": ["spatial_relations", "ifc_class"],
            # Fires ONLY when predicate == "CONTINUOUS"
            "predicate_filter": "CONTINUOUS",
            "description": "CONTINUOUS predicate — element spans multiple storeys (property filter, ~1-5 candidates)",
            "template_memory": None,
            "template_cypher": """
                MATCH (target:IFCElement)
                WHERE target.ifc_type = $subject_type
                  AND target.is_continuous = true
                  AND target.ifc_model = $model
                  AND toLower(target.top_constraint) CONTAINS toLower($top_storey)
                RETURN target.guid as guid, target.name as name, target.ifc_type as type,
                       target.base_constraint as ref_storey,
                       target.top_constraint  as ref_type
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
                  AND (e.ifc_type = $type OR e.ifc_type STARTS WITH $type)
                  AND e.ifc_model = $model
                RETURN e.guid as guid, e.name as name, e.ifc_type as type,
                       sp.name as space
            """
        },
        # 6.1.0: priority-2 `name_keyword` strategy removed. Vocabulary mismatch
        # (GT-label ↔ IFC-name CONTAINS = 0/31) made it signal-dead. Field is
        # now rerank-only descriptor; see `_structured_evidence` in
        # graph_rag_rerank_ap.py.
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
                  AND (e.ifc_type = $type OR e.ifc_type STARTS WITH $type)
                  AND e.ifc_model = $model
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
                  AND e.ifc_model = $model
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
                WHERE (e.ifc_type = $type OR e.ifc_type STARTS WITH $type)
                  AND e.ifc_model = $model
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
                WHERE e.ifc_model = $model
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

        # Route CONTINUOUS vs edge-traversal predicates to the correct rule
        predicate_filter = rule.get("predicate_filter")   # rule only fires for this predicate
        predicate_exclude = rule.get("predicate_exclude", [])  # rule skips these predicates
        if predicate_filter or predicate_exclude:
            triplet_predicate = (
                constraints.spatial_relations[0].predicate
                if constraints.spatial_relations else None
            )
            if predicate_filter and triplet_predicate != predicate_filter:
                return False
            if predicate_exclude and triplet_predicate in predicate_exclude:
                return False

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

        # Phase 4 new field mappings
        if "space_name" in required_fields:
            params["space_name"] = constraints.space_name

        # 6.1.0: target_name_keyword no longer routed to any retrieval strategy.

        # Phase 5: spatial_triplet — extract predicate data from the first triplet
        if "spatial_relations" in required_fields and constraints.spatial_relations:
            triplet: SpatialTriplet = constraints.spatial_relations[0]
            params["subject_type"] = triplet.subject_type
            params["predicate"] = triplet.predicate
            params["object_type"] = triplet.object_type
            params["spatial_relations"] = [t.model_dump() for t in constraints.spatial_relations]
            if triplet.object_material:
                params["object_material"] = triplet.object_material
            if any(t.direction for t in constraints.spatial_relations):
                params["has_directional_fingerprint"] = True
            if any(t.object_subtype for t in constraints.spatial_relations):
                params["has_object_subtype_fingerprint"] = True
            # storey routing depends on context:
            # - H2 eval sets storey_name = top_constraint for CONTINUOUS
            # - LoRA sets storey_name = base storey (the floor the element is on)
            # Strategy: put storey_name in BOTH params; Cypher uses OR logic
            # so either match works. The more specific match wins.
            if "storey" not in params:
                params["storey"] = constraints.storey_name or ""
            params["top_storey"] = constraints.storey_name or ""
            params["chain_mode"] = (
                "multi_chain_anchor"
                if len(constraints.spatial_relations) >= 2
                else "single_chain"
            )

        if constraints.position_context:
            params["position_context"] = constraints.position_context
            if constraints.position_context_confidence is not None:
                params["position_context_confidence"] = constraints.position_context_confidence
            if constraints.position_context_source:
                params["position_context_source"] = constraints.position_context_source
            parsed_position = self._parse_position_context(constraints.position_context)
            should_hard_filter = (
                parsed_position
                and (
                    constraints.position_context_confidence is None
                    or constraints.position_context_confidence >= 0.8
                )
            )
            if should_hard_filter:
                params.update(parsed_position)

        if params.get("position_index") is not None:
            params["fingerprint_level_requested"] = "exact_slot"
        elif params.get("has_object_subtype_fingerprint") or params.get("has_directional_fingerprint"):
            params["fingerprint_level_requested"] = "relation_fingerprint"
        elif params.get("spatial_relations"):
            params["fingerprint_level_requested"] = "topology_only"
        else:
            params["fingerprint_level_requested"] = "attribute_only"

        # 6.1.0: target_name_keyword is no longer routed into retrieval params
        # (signal-dead post-filter; consumed only by the graph-RAG reranker).
        # 6.1.1: size_cluster is propagated, but the backend's behaviour is
        # mode-controlled (off/soft/hard) via config — see `size_cluster_mode`.
        if constraints.size_cluster:
            params["target_size_cluster"] = constraints.size_cluster
        # 6.1.6: size_band is propagated alongside; backend uses STARTS WITH
        # against target.size_cluster on the candidate side. Mode-controlled
        # via `size_band_mode` (off/soft/hard) parallel to size_cluster_mode.
        if constraints.size_band:
            params["target_size_band"] = constraints.size_band

        return params

    def _parse_position_context(self, value: str) -> Dict[str, int]:
        """Parse human-readable slot text into planner-friendly integers."""
        if not value:
            return {}
        match = self._POSITION_CONTEXT_RE.search(value.strip())
        if not match:
            return {}
        index = int(match.group("index"))
        total = int(match.group("total"))
        return {
            "position_index": index,
            "position_total": total,
        }

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
            "spatial_triplet":  3,  # Most specific: topological edge → ~1-3 matches
            "continuous_span":  5,  # Property filter on is_continuous + top_constraint
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
    _POSITION_CONTEXT_RE = re.compile(
        r"(?P<index>\d+)(?:st|nd|rd|th)\s+of\s+(?P<total>\d+)\s+openings",
        flags=re.IGNORECASE,
    )
