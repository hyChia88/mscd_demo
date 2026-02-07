"""
Template-Based Query Planner

Translates extracted constraints into deterministic, priority-ordered query plans.
No LLM generation - uses fixed templates for reproducibility.
"""

from typing import List, Dict, Any
from .types import Constraints, QueryPlan


class QueryPlanner:
    """
    Deterministic query planning with priority-based fallbacks.

    Generates ordered list of query plans from constraints, trying
    most specific queries first and falling back to broader queries.
    """

    # Priority rules: ordered from most specific to most general
    PRIORITY_RULES = [
        {
            "priority": 1,
            "strategy": "storey+type",
            "requires": ["storey_name", "ifc_class"],
            "description": "Most specific: both storey and IFC type known",
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
            "priority": 2,
            "strategy": "storey_only",
            "requires": ["storey_name"],
            "description": "Narrow to storey/floor only",
            "template_memory": "filter_by_storey",
            "template_cypher": """
                MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
                WHERE toLower(s.name) CONTAINS toLower($storey)
                RETURN e.guid as guid, e.name as name, e.ifc_type as type,
                       s.name as storey
            """
        },
        {
            "priority": 3,
            "strategy": "type_only",
            "requires": ["ifc_class"],
            "description": "Filter by IFC type across all storeys",
            "template_memory": "filter_by_type",
            "template_cypher": """
                MATCH (e:IFCElement)
                WHERE e.ifc_type = $type
                RETURN e.guid as guid, e.name as name, e.ifc_type as type
            """
        },
        {
            "priority": 4,
            "strategy": "keyword",
            "requires": ["near_keywords"],
            "description": "Text search using spatial keywords",
            "template_memory": "search_by_keywords",
            "template_cypher": """
                MATCH (e:IFCElement)
                WHERE toLower(e.name) CONTAINS toLower($keyword)
                   OR toLower(e.description) CONTAINS toLower($keyword)
                RETURN e.guid as guid, e.name as name, e.ifc_type as type
            """
        },
        {
            "priority": 5,
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
                priority=5,
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
            "storey+type": 50,    # Very specific - small pool
            "storey_only": 200,   # All elements on one floor
            "type_only": 150,     # All elements of one type
            "keyword": 100,       # Keyword search - variable
            "fallback": 100       # Capped at 100
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
