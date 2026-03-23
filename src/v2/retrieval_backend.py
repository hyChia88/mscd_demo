"""
Unified Retrieval Backend

Provides a unified interface for memory/neo4j/+clip retrieval modes,
reusing v1 components (IFCEngine, VisualAligner).
"""

from typing import List, Dict, Any, Optional
from .types import QueryPlan, RetrievalResult


class RetrievalBackend:
    """
    Unified retrieval interface for memory/neo4j/+clip modes.

    Executes query plans and optionally applies CLIP reranking.
    """

    def __init__(
        self,
        engine: Any,  # IFCEngine from v1
        retrieval_mode: str,  # "memory" or "neo4j"
        visual_aligner: Optional[Any] = None,  # VisualAligner from v1
        use_clip: bool = False,
        p0_strategy: str = "p0_union_p1",  # retrieval strategy for P0
    ):
        """
        Initialize retrieval backend.

        Args:
            engine: IFCEngine instance from v1
            retrieval_mode: "memory" or "neo4j"
            visual_aligner: VisualAligner instance (optional)
            use_clip: Whether to enable CLIP reranking
            p0_strategy: How to handle P0 spatial queries:
                "p0_only"          — P0 fires alone, no safety net (original)
                "p1_only"          — Skip P0, always use storey+type
                "p0_intersect_p1"  — P0 ∩ P1 intersection (aggressive)
                "p0_union_p1"      — P0 ∪ P1 union (default — best recall+ranking)
        """
        self.engine = engine
        self.retrieval_mode = retrieval_mode
        self.visual_aligner = visual_aligner
        self.use_clip = use_clip and (visual_aligner is not None)
        self.p0_strategy = p0_strategy

    async def execute_plan(
        self,
        plan: QueryPlan,
        image_paths: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        Execute a query plan and return candidates.

        Args:
            plan: QueryPlan to execute
            image_paths: Optional images for CLIP reranking

        Returns:
            RetrievalResult with candidates
        """
        # Reset per-call fallback tracking state
        self._fallback_triggered = False
        self._strategy_actually_used = plan.strategy

        # Step 1: Execute base query
        if self.retrieval_mode == "neo4j":
            candidates = self._execute_neo4j(plan)
        else:
            candidates = self._execute_memory(plan)

        # Step 1b: P0 strategy — controls how spatial queries interact with P1
        #   p0_only:          keep P0 result as-is (original behavior)
        #   p1_only:          discard P0, use storey+type instead
        #   p0_intersect_p1:  P0 ∩ P1 (defensive, default)
        #   p0_union_p1:      P0 ∪ P1 (max recall)
        if plan.strategy in ("spatial_triplet", "continuous_span"):
            if self.p0_strategy == "p1_only":
                # Skip P0 entirely — use storey+type
                p1_candidates = self._get_storey_type_pool(plan.params)
                if p1_candidates:
                    candidates = p1_candidates
                    self._fallback_triggered = True
                    self._strategy_actually_used = "storey+type"

            elif self.p0_strategy == "p0_intersect_p1":
                p0_guids = {c["guid"] for c in candidates}
                p1_candidates = self._get_storey_type_pool(plan.params)
                if p1_candidates:
                    p1_guids = {c["guid"] for c in p1_candidates}
                    intersection = [c for c in candidates if c["guid"] in p1_guids]
                    if intersection:
                        candidates = intersection
                        self._strategy_actually_used = f"{plan.strategy}∩storey+type"
                    elif not p0_guids:
                        candidates = p1_candidates
                        self._fallback_triggered = True
                        self._strategy_actually_used = "storey+type"

            elif self.p0_strategy == "p0_union_p1":
                p0_guids = {c["guid"] for c in candidates}
                p1_candidates = self._get_storey_type_pool(plan.params)
                if p1_candidates:
                    # Union: P0 first (topology-boosted), then P1-only elements
                    p1_only = [c for c in p1_candidates if c["guid"] not in p0_guids]
                    candidates = candidates + p1_only
                    self._strategy_actually_used = f"{plan.strategy}∪storey+type"

            # else: p0_only — keep P0 result as-is (no modification)

        # Step 2: Apply CLIP reranking if enabled and images available
        rerank_applied = False
        backend_name = self.retrieval_mode

        if self.use_clip and image_paths and self.visual_aligner and candidates:
            # Store original order for comparison
            original_order = [c["guid"] for c in candidates]

            # Apply CLIP reranking
            candidates = self._rerank_with_clip(candidates, image_paths)

            # Check if order changed
            reranked_order = [c["guid"] for c in candidates]
            rerank_applied = (original_order != reranked_order)

            backend_name = f"{self.retrieval_mode}+clip"

        return RetrievalResult(
            candidates=candidates,
            pool_size=len(candidates),
            query_plan_used=plan,
            backend=backend_name,
            rerank_applied=rerank_applied,
            fallback_triggered=self._fallback_triggered,
            strategy_actually_used=self._strategy_actually_used
        )

    def _execute_memory(self, plan: QueryPlan) -> List[Dict[str, Any]]:
        """
        Execute plan using in-memory spatial index.

        Uses IFCEngine's spatial_index for fast lookups.

        Args:
            plan: QueryPlan to execute

        Returns:
            List of candidate elements
        """
        strategy = plan.strategy
        params = plan.params

        # ── Priority 0 strategies: no topology in memory mode — degrade ─────
        if strategy == "spatial_triplet":
            # No edge traversal available in memory mode.
            # Degrade to storey+type (subject_type + storey), or type_only.
            subject_type = params.get("subject_type", "")
            storey = params.get("storey", "")
            if subject_type and storey:
                fallback_strat = "storey+type"
                fallback = QueryPlan(
                    priority=4, strategy=fallback_strat,
                    params={"storey": storey, "type": subject_type},
                    expected_pool_size=50
                )
            else:
                fallback_strat = "type_only"
                fallback = QueryPlan(
                    priority=6, strategy=fallback_strat,
                    params={"type": subject_type},
                    expected_pool_size=150
                )
            self._fallback_triggered = True
            self._strategy_actually_used = fallback_strat
            return self._execute_memory(fallback)

        elif strategy == "continuous_span":
            # No property graph in memory mode — degrade to type_only.
            subject_type = params.get("subject_type", "")
            self._fallback_triggered = True
            self._strategy_actually_used = "type_only"
            fallback = QueryPlan(
                priority=6, strategy="type_only",
                params={"type": subject_type},
                expected_pool_size=150
            )
            return self._execute_memory(fallback)

        # ── Phase 5A: new high-priority strategies ──────────────────────────
        if strategy == "space+type":
            # Most specific: elements in a named room filtered by type
            space_key = (params.get("space_name") or "").lower()
            target_type = params.get("type") or ""
            results = self.engine.find_elements_in_space(space_key)
            filtered = [r for r in results if r.get("type") == target_type]
            # Graceful degradation: room found but type filter removes all → return unfiltered
            return filtered if filtered else results

        elif strategy == "name_keyword":
            # Equipment brand/ID fuzzy name match
            keyword = params.get("name_keyword") or ""
            return self.engine.query_elements_by_name_keyword(keyword)

        elif strategy == "neighbor+type":
            # Topology requires graph — memory mode has no adjacency data.
            # Degrade to type_only so caller still gets a useful candidate pool.
            type_only_plan = QueryPlan(
                priority=5,
                strategy="type_only",
                params={"type": params.get("type", "")},
                expected_pool_size=150
            )
            return self._execute_memory(type_only_plan)

        # ── Original strategies ─────────────────────────────────────────────
        elif strategy == "storey+type":
            # Most specific: filter by storey AND type
            storey_key = (params.get("storey") or "").lower()
            target_type = params.get("type") or ""

            results = self.engine.find_elements_in_space(storey_key)
            return [r for r in results if r.get("type") == target_type]

        elif strategy == "storey_only":
            # Filter by storey only
            storey_key = (params.get("storey") or "").lower()
            return self.engine.find_elements_in_space(storey_key)

        elif strategy == "type_only":
            # Search across all storeys for specific type
            target_type = params.get("type", "")
            all_results = []

            for space_elements in self.engine.spatial_index.values():
                all_results.extend([
                    e for e in space_elements if e.get("type") == target_type
                ])

            return all_results

        elif strategy == "keyword":
            # Text search using keywords
            keywords = params.get("keywords", [params.get("keyword", "")])
            all_results = []

            for space_elements in self.engine.spatial_index.values():
                for element in space_elements:
                    name = (element.get("name") or "").lower()
                    desc = (element.get("description") or "").lower()

                    # Check if any keyword matches
                    for keyword in keywords:
                        if not keyword:
                            continue
                        if keyword.lower() in name or keyword.lower() in desc:
                            all_results.append(element)
                            break

            return all_results

        elif strategy == "fallback":
            # Return first 100 elements across all spaces
            all_results = []
            for space_elements in self.engine.spatial_index.values():
                all_results.extend(space_elements)
                if len(all_results) >= 100:
                    break

            return all_results[:100]

        return []

    def _execute_neo4j(self, plan: QueryPlan) -> List[Dict[str, Any]]:
        """
        Execute plan using Neo4j graph queries.

        Uses IFCEngine's Neo4j connection for graph-based retrieval.

        Args:
            plan: QueryPlan to execute

        Returns:
            List of candidate elements
        """
        strategy = plan.strategy
        params = plan.params

        # ── Priority 0 strategies: direct Cypher on topology edges/properties ─
        if strategy == "spatial_triplet":
            # Edge-traversal for FILLS / ADJACENT_TO / ON_TOP_OF / etc.
            # predicate is a validated Literal — safe for f-string injection.
            if not self.engine.neo4j_conn:
                return self._execute_memory(plan)
            subject_type    = params.get("subject_type", "")
            predicate       = params.get("predicate", "")
            object_type     = params.get("object_type", "")
            storey          = params.get("storey", "")
            object_material = params.get("object_material", "")
            if not predicate:
                return []
            # Resolve storey → all canonical siblings sharing the same floor number.
            # "Level 1" and "1 - First Floor" are different IFC storeys but same floor_num=1.
            # Elements are split across both → must match either.
            storey_siblings = self._resolve_storey_siblings(storey) if storey else []

            # ── 2-hop OPTIONAL MATCH detection ───────────────────────────────
            # If spatial_relations has 2+ triplets, triplet[0] is the hard
            # filter (hop 1) and triplet[1] is the soft re-rank (hop 2).
            # The OPTIONAL MATCH NEVER reduces the candidate pool vs single-hop
            # — it only reorders (has_hop2=true ranked first).
            spatial_rels = params.get("spatial_relations", [])
            has_hop2 = len(spatial_rels) >= 2

            if has_hop2:
                hop2 = spatial_rels[1]
                predicate2 = hop2.get("predicate", "")
                object_type2 = hop2.get("object_type", "")
            else:
                predicate2 = ""
                object_type2 = ""

            # IFC subtype-aware match: IfcWall matches IfcWallStandardCase etc.
            # T1.3: Added graceful material filter on ref element.
            model_key = self.engine.model_key
            if has_hop2 and predicate2:
                # 2-hop Cypher: hard filter on hop 1, soft re-rank on hop 2
                cypher = f"""
                    MATCH (target:IFCElement)-[:{predicate}]->(ref:IFCElement)
                    WHERE (target.ifc_type = $subject_type
                           OR target.ifc_type STARTS WITH $subject_type)
                      AND (ref.ifc_type = $object_type
                           OR ref.ifc_type STARTS WITH $object_type)
                      AND target.ifc_model = $model
                      AND (size($storey_list) = 0
                           OR ANY(s IN $storey_list WHERE toLower(target.storey) CONTAINS s))
                      AND ($object_material = ''
                           OR toLower(ref.material) CONTAINS toLower($object_material))
                    OPTIONAL MATCH (ref)-[:{predicate2}]->(ref2:IFCElement)
                    WHERE (ref2.ifc_type = $object_type2
                           OR ref2.ifc_type STARTS WITH $object_type2)
                    RETURN DISTINCT target.guid AS guid, target.name AS name,
                           target.ifc_type AS type,
                           ref.ifc_type AS ref_type, ref.storey AS ref_storey,
                           ref2 IS NOT NULL AS has_hop2
                    ORDER BY has_hop2 DESC
                """
                result = self.engine.neo4j_conn.run(
                    cypher,
                    subject_type=subject_type,
                    object_type=object_type,
                    storey_list=storey_siblings,
                    object_material=object_material,
                    object_type2=object_type2,
                    model=model_key,
                )
            else:
                # Single-hop Cypher (original behavior)
                cypher = f"""
                    MATCH (target:IFCElement)-[:{predicate}]->(ref:IFCElement)
                    WHERE (target.ifc_type = $subject_type
                           OR target.ifc_type STARTS WITH $subject_type)
                      AND (ref.ifc_type = $object_type
                           OR ref.ifc_type STARTS WITH $object_type)
                      AND target.ifc_model = $model
                      AND (size($storey_list) = 0
                           OR ANY(s IN $storey_list WHERE toLower(target.storey) CONTAINS s))
                      AND ($object_material = ''
                           OR toLower(ref.material) CONTAINS toLower($object_material))
                    RETURN DISTINCT target.guid as guid, target.name as name,
                           target.ifc_type as type,
                           ref.ifc_type as ref_type, ref.storey as ref_storey
                """
                result = self.engine.neo4j_conn.run(
                    cypher,
                    subject_type=subject_type,
                    object_type=object_type,
                    storey_list=storey_siblings,
                    object_material=object_material,
                    model=model_key,
                )
            candidates = [dict(r) for r in result]

            # Predicate relaxation step 1: if storey filter gives 0, retry without storey
            # (stays Priority-0, avoids collapsing to storey+type prematurely)
            if not candidates and storey_siblings:
                if has_hop2 and predicate2:
                    cypher_no_storey = f"""
                        MATCH (target:IFCElement)-[:{predicate}]->(ref:IFCElement)
                        WHERE (target.ifc_type = $subject_type
                               OR target.ifc_type STARTS WITH $subject_type)
                          AND (ref.ifc_type = $object_type
                               OR ref.ifc_type STARTS WITH $object_type)
                          AND target.ifc_model = $model
                          AND ($object_material = ''
                               OR toLower(ref.material) CONTAINS toLower($object_material))
                        OPTIONAL MATCH (ref)-[:{predicate2}]->(ref2:IFCElement)
                        WHERE (ref2.ifc_type = $object_type2
                               OR ref2.ifc_type STARTS WITH $object_type2)
                        RETURN DISTINCT target.guid AS guid, target.name AS name,
                               target.ifc_type AS type,
                               ref.ifc_type AS ref_type, ref.storey AS ref_storey,
                               ref2 IS NOT NULL AS has_hop2
                        ORDER BY has_hop2 DESC
                    """
                    result2 = self.engine.neo4j_conn.run(
                        cypher_no_storey,
                        subject_type=subject_type,
                        object_type=object_type,
                        object_material=object_material,
                        object_type2=object_type2,
                        model=model_key,
                    )
                else:
                    cypher_no_storey = f"""
                        MATCH (target:IFCElement)-[:{predicate}]->(ref:IFCElement)
                        WHERE (target.ifc_type = $subject_type
                               OR target.ifc_type STARTS WITH $subject_type)
                          AND (ref.ifc_type = $object_type
                               OR ref.ifc_type STARTS WITH $object_type)
                          AND target.ifc_model = $model
                          AND ($object_material = ''
                               OR toLower(ref.material) CONTAINS toLower($object_material))
                        RETURN DISTINCT target.guid as guid, target.name as name,
                               target.ifc_type as type,
                               ref.ifc_type as ref_type, ref.storey as ref_storey
                    """
                    result2 = self.engine.neo4j_conn.run(
                        cypher_no_storey,
                        subject_type=subject_type,
                        object_type=object_type,
                        object_material=object_material,
                        model=model_key,
                    )
                candidates = [dict(r) for r in result2]
            # Predicate relaxation step 2: no topology edges at all → storey+type
            if not candidates:
                fallback_strategy = "storey+type" if storey else "type_only"
                self._fallback_triggered = True
                self._strategy_actually_used = fallback_strategy
                fallback_params = {"storey": storey, "type": subject_type} if storey \
                    else {"type": subject_type}
                fallback_priority = 4 if storey else 6
                fallback_pool = 50 if storey else 150
                return self._execute_neo4j(QueryPlan(
                    priority=fallback_priority,
                    strategy=fallback_strategy,
                    params=fallback_params,
                    expected_pool_size=fallback_pool
                ))
            # T1.1: Post-filter by target_name_keyword (e.g. BALANS 15M window subtype)
            return self._post_filter_by_name_keyword(candidates, params)

        elif strategy == "continuous_span":
            # Property-filter for CONTINUOUS (is_continuous + top_constraint).
            if not self.engine.neo4j_conn:
                return self._execute_memory(plan)
            subject_type = params.get("subject_type", "")
            top_storey   = params.get("top_storey", "")
            storey       = params.get("storey", "")
            # Resolve to sibling lists for naming-agnostic matching.
            top_siblings = self._resolve_storey_siblings(top_storey)
            storey_siblings = self._resolve_storey_siblings(storey)
            # storey_name could be base storey OR top_constraint (context-dependent).
            # Use OR: match if EITHER target.storey or target.top_constraint
            # matches any sibling. Handles both LoRA output and H2 eval.
            cypher = """
                MATCH (target:IFCElement)
                WHERE target.ifc_type = $subject_type
                  AND target.is_continuous = true
                  AND target.ifc_model = $model
                  AND (
                       (size($top_list) = 0 AND size($storey_list) = 0)
                    OR ANY(s IN $top_list WHERE toLower(target.top_constraint) CONTAINS s)
                    OR ANY(s IN $storey_list WHERE toLower(target.storey) CONTAINS s)
                  )
                RETURN target.guid as guid, target.name as name,
                       target.ifc_type as type,
                       target.base_constraint as ref_storey,
                       target.top_constraint  as ref_type
            """
            result = self.engine.neo4j_conn.run(
                cypher,
                subject_type=subject_type,
                top_list=top_siblings,
                storey_list=storey_siblings,
                model=self.engine.model_key,
            )
            candidates = [dict(r) for r in result]
            # Predicate relaxation: 0 results → fall back to type_only
            if not candidates:
                self._fallback_triggered = True
                self._strategy_actually_used = "type_only"
                return self._execute_neo4j(QueryPlan(
                    priority=6,
                    strategy="type_only",
                    params={"type": subject_type},
                    expected_pool_size=150
                ))
            # T1.1: Post-filter by target_name_keyword
            return self._post_filter_by_name_keyword(candidates, params)

        # ── Phase 5B: new high-priority strategies (Neo4j graph queries) ──────
        elif strategy == "space+type":
            # IFCSpace-level query via Phase 1a method
            space_name = params.get("space_name", "")
            target_type = params.get("type", "")
            return self.engine.query_elements_in_space(space_name, ifc_type=target_type)

        elif strategy == "name_keyword":
            # Fuzzy name match via Phase 1a method
            keyword = params.get("name_keyword", "")
            return self.engine.query_elements_by_name_keyword(keyword)

        elif strategy == "neighbor+type":
            # Topological adjacency via Phase 1a method (Neo4j only)
            neighbor_type = params.get("neighbor_type", "")
            target_type = params.get("type", "")
            return self.engine.query_elements_by_neighbor(target_type, neighbor_type)

        # ── Original strategies ─────────────────────────────────────────────
        elif strategy == "storey+type":
            # Use IFCEngine's query_elements_by_level with type filter
            storey_name = params.get("storey", "")
            target_type = params.get("type", "")

            # Call v1 IFCEngine method
            results = self.engine.query_elements_by_level(storey_name)

            # Filter by type
            return [r for r in results if r.get("type") == target_type or r.get("ifc_type") == target_type]

        elif strategy == "storey_only":
            # Query all elements on storey
            storey_name = params.get("storey", "")
            return self.engine.query_elements_by_level(storey_name)

        elif strategy == "type_only":
            # Query by type across all storeys
            # Note: IFCEngine may not have a direct method for this
            # Fall back to memory-based retrieval for now
            return self._execute_memory(plan)

        elif strategy == "keyword":
            # Fall back to memory for keyword search
            return self._execute_memory(plan)

        elif strategy == "fallback":
            # Fall back to memory
            return self._execute_memory(plan)

        return []

    def _post_filter_by_name_keyword(
        self,
        candidates: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        T1.1 — Post-filter P0 candidates by target_name_keyword.

        Window ObjectType (e.g. BALANS 15M vs 10M) is discriminating (pool 42→~9).
        Applied after Cypher returns candidates to avoid Cypher complexity.
        Graceful: never filters to empty set; skips if pool already small.
        """
        keyword = params.get("target_name_keyword", "")
        if not keyword or len(candidates) <= 3:
            return candidates
        kw_lower = keyword.lower()
        filtered = [
            c for c in candidates
            if kw_lower in (c.get("name") or "").lower()
        ]
        # Graceful: don't filter to empty
        return filtered if filtered else candidates

    def _get_storey_type_pool(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run a storey+type query using the same params as the P0 plan.
        Used by the P0 ∩ P1 defensive intersection.
        Returns [] if storey or subject_type is missing.
        """
        storey = params.get("storey", "")
        subject_type = params.get("subject_type", "")
        if not storey or not subject_type:
            return []
        p1_plan = QueryPlan(
            priority=4,
            strategy="storey+type",
            params={"storey": storey, "type": subject_type},
            expected_pool_size=50,
        )
        if self.retrieval_mode == "neo4j":
            return self._execute_neo4j(p1_plan)
        return self._execute_memory(p1_plan)

    def _resolve_storey_siblings(self, storey_query: str) -> list:
        """
        Resolve a storey query → list of ALL canonical siblings sharing the same
        floor_num. Delegates to IFCEngine._resolve_storey_query() which now
        returns siblings directly.

        e.g. "Floor 1" → ["level 1", "1 - first floor"]

        This handles IFC naming heterogeneity: different modelers use different
        naming conventions, and the same physical floor may appear under multiple
        IfcBuildingStorey names. Returns [] for empty query.
        """
        if not storey_query:
            return []
        resolver = getattr(self.engine, "_resolve_storey_query", None)
        if resolver is None:
            return [storey_query.lower()]
        resolved = resolver(storey_query.lower())
        if isinstance(resolved, list) and resolved:
            return resolved
        return [storey_query.lower()]

    def _apply_property_filter(
        self,
        candidates: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Phase 5C — Post-filter candidates by Pset property values.

        Only runs when params contains a recognised property constraint key
        (firerating, loadbearing, isexternal, material, acousticrating).
        Currently not triggered — reserved for future Constraints fields.

        Graceful degrade: if filter removes all candidates, return original list.
        """
        _PROP_KEYS = ("firerating", "loadbearing", "isexternal",
                      "material", "acousticrating")
        active = {k: params[k] for k in _PROP_KEYS if k in params and params[k] is not None}
        if not active:
            return candidates  # no property constraint — skip entirely

        filtered = []
        for c in candidates:
            props = self.engine.get_element_properties(c["guid"])
            flat = {}
            for pset in props.get("PropertySets", {}).values():
                for k, v in pset.items():
                    flat[k.lower()] = v
            if all(flat.get(k) == v for k, v in active.items()):
                filtered.append(c)

        return filtered if filtered else candidates

    def _rerank_with_clip(
        self,
        candidates: List[Dict[str, Any]],
        image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using CLIP visual similarity.

        Uses VisualAligner from v1 for image-to-element matching.

        Args:
            candidates: Original candidate list
            image_paths: Paths to site images

        Returns:
            Reranked candidate list
        """
        if not image_paths or not self.visual_aligner or not candidates:
            return candidates

        try:
            # Use first image for matching (can be extended to multiple)
            matches = self.visual_aligner.match_image_to_elements(
                image_paths[0],
                candidates,
                top_k=len(candidates)
            )

            # Rebuild candidates list in new rank order
            guid_to_candidate = {c.get("guid"): c for c in candidates}
            reranked = []

            for match in matches:
                guid = match.get("guid")
                if guid in guid_to_candidate:
                    candidate = guid_to_candidate[guid]
                    # Add CLIP score to candidate
                    candidate["clip_score"] = match.get("score", 0.0)
                    reranked.append(candidate)

            # Add any candidates that weren't matched (shouldn't happen)
            for candidate in candidates:
                if candidate["guid"] not in [c["guid"] for c in reranked]:
                    candidate["clip_score"] = 0.0
                    reranked.append(candidate)

            return reranked

        except Exception as e:
            print(f"⚠️  CLIP reranking failed: {e}")
            # Return original order on error
            return candidates
