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
        size_cluster_mode: str = "soft",  # "off" | "soft" | "hard"
        size_band_mode: str = "hard",     # "off" | "soft" | "hard" — matches via STARTS WITH
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
            size_cluster_mode: How target.size_cluster influences spatial_triplet:
                "off"  — ignore (rerank consumes it)
                "soft" — ORDER BY match DESC, pos (preference, never excludes)
                "hard" — WHERE target.size_cluster = $val (only safe at ≥70% precision)
        """
        self.engine = engine
        self.retrieval_mode = retrieval_mode
        self.visual_aligner = visual_aligner
        self.use_clip = use_clip and (visual_aligner is not None)
        self.p0_strategy = p0_strategy
        if size_cluster_mode not in ("off", "soft", "hard"):
            raise ValueError(f"size_cluster_mode must be off/soft/hard, got {size_cluster_mode!r}")
        if size_band_mode not in ("off", "soft", "hard"):
            raise ValueError(f"size_band_mode must be off/soft/hard, got {size_band_mode!r}")
        self.size_cluster_mode = size_cluster_mode
        self.size_band_mode = size_band_mode

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

        raw_pool_size = len(candidates)
        if plan.strategy in ("spatial_triplet", "continuous_span"):
            actual = self._strategy_actually_used or plan.strategy
            # Treat topology→attribute demotions as zero raw-topology hits so the
            # demo can show the pure P0 pool before the P1 safety-net kicks in.
            if actual in ("storey+type", "type_only"):
                raw_pool_size = 0

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
            raw_pool_size=raw_pool_size,
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

            # ── Phase 3: multi-anchor vs single-hop dispatch ──────────────────
            # 2+ spatial_relations → AND hard-filter with same-wall constraint.
            # 1 spatial_relation  → original single-hop Cypher (unchanged).
            spatial_rels = params.get("spatial_relations", [])
            model_key = self.engine.model_key

            if self._requires_multi_chain(spatial_rels, params):
                # Default planner path: full fingerprint first, then controlled fallbacks.
                candidates = self._execute_multi_anchor(
                    subject_type,
                    spatial_rels,
                    storey_siblings,
                    object_material,
                    params,
                    include_relation_fingerprint=True,
                    include_position=True,
                )
                if candidates and self._has_relation_fingerprint(spatial_rels, params):
                    self._strategy_actually_used = f"{plan.strategy}[full_fingerprint]"

                # Exact-slot relaxation
                if not candidates and params.get("position_index") is not None:
                    candidates = self._execute_multi_anchor(
                        subject_type,
                        spatial_rels,
                        storey_siblings,
                        object_material,
                        params,
                        include_relation_fingerprint=True,
                        include_position=False,
                    )
                    if candidates:
                        self._fallback_triggered = True
                        self._strategy_actually_used = f"{plan.strategy}[no_position]"

                # Relation fingerprint relaxation
                if not candidates and self._has_relation_fingerprint(spatial_rels, params):
                    candidates = self._execute_multi_anchor(
                        subject_type,
                        spatial_rels,
                        storey_siblings,
                        object_material,
                        params,
                        include_relation_fingerprint=False,
                        include_position=False,
                    )
                    if candidates:
                        self._fallback_triggered = True
                        self._strategy_actually_used = f"{plan.strategy}[topology_only]"

                # Storey relaxation: retry without storey filter
                if not candidates and storey_siblings:
                    candidates = self._execute_multi_anchor(
                        subject_type,
                        spatial_rels,
                        [],
                        object_material,
                        params,
                        include_relation_fingerprint=False,
                        include_position=False,
                    )
                    if candidates:
                        self._fallback_triggered = True
                        self._strategy_actually_used = f"{plan.strategy}[no_storey]"

                # Relation relaxation: drop weakest SR one at a time
                if not candidates:
                    candidates = self._relax_multi_anchor(
                        subject_type,
                        spatial_rels,
                        storey_siblings,
                        object_material,
                        params,
                        include_relation_fingerprint=False,
                        include_position=False,
                    )
                    if candidates:
                        self._fallback_triggered = True
                        self._strategy_actually_used = f"{plan.strategy}[relaxed]"
            else:
                # 6.1.1: size_cluster handling is mode-controlled.
                #   off  → ignore (no clause)
                #   soft → ORDER BY match DESC, pos
                #   hard → WHERE target.size_cluster = $val (legacy G9 mode)
                # 6.1.6: size_band uses STARTS WITH on the candidate's
                # full size_cluster property (no Neo4j migration needed).
                # When BOTH size_band and size_cluster are set, size_band
                # is preferred (broader, ResNet-derived) and size_cluster
                # is ignored to avoid double-narrowing.
                # NOTE: ORDER BY must reference projected aliases under
                # RETURN DISTINCT — Cypher rejects bare variable refs there.
                size_cluster_val = params.get("target_size_cluster")
                size_band_val = params.get("target_size_band")
                size_where = ""
                size_order = "ORDER BY pos"
                size_params: Dict[str, Any] = {}
                if size_band_val and self.size_band_mode != "off":
                    # Prefix-match against candidate's size_cluster (band_<dims>).
                    size_params["target_size_band_prefix"] = f"{size_band_val}_"
                    if self.size_band_mode == "hard":
                        size_where = (
                            "\n                      AND target.size_cluster"
                            " STARTS WITH $target_size_band_prefix"
                        )
                    else:  # soft
                        size_order = (
                            "ORDER BY CASE WHEN size_cluster STARTS WITH"
                            " $target_size_band_prefix THEN 0 ELSE 1 END, pos"
                        )
                elif size_cluster_val and self.size_cluster_mode != "off":
                    size_params["target_size_cluster"] = str(size_cluster_val)
                    if self.size_cluster_mode == "hard":
                        size_where = "\n                      AND target.size_cluster = $target_size_cluster"
                    else:  # soft
                        size_order = (
                            "ORDER BY CASE WHEN size_cluster = $target_size_cluster"
                            " THEN 0 ELSE 1 END, pos"
                        )

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
                           OR toLower(ref.material) CONTAINS toLower($object_material)){size_where}
                    RETURN DISTINCT target.guid as guid, target.name as name,
                           target.ifc_type as type,
                           target.size_cluster as size_cluster,
                           ref.ifc_type as ref_type, ref.storey as ref_storey,
                           coalesce(target.wall_position_index, 999999) as pos
                    {size_order}
                """
                result = self.engine.neo4j_conn.run(
                    cypher,
                    subject_type=subject_type,
                    object_type=object_type,
                    storey_list=storey_siblings,
                    object_material=object_material,
                    model=model_key,
                    **size_params,
                )
                candidates = [dict(r) for r in result]

                # Storey relaxation: no results → retry without storey
                if not candidates and storey_siblings:
                    cypher_no_storey = f"""
                        MATCH (target:IFCElement)-[:{predicate}]->(ref:IFCElement)
                        WHERE (target.ifc_type = $subject_type
                               OR target.ifc_type STARTS WITH $subject_type)
                          AND (ref.ifc_type = $object_type
                               OR ref.ifc_type STARTS WITH $object_type)
                          AND target.ifc_model = $model
                          AND ($object_material = ''
                               OR toLower(ref.material) CONTAINS toLower($object_material)){size_where}
                        RETURN DISTINCT target.guid as guid, target.name as name,
                               target.ifc_type as type,
                               target.size_cluster as size_cluster,
                               ref.ifc_type as ref_type, ref.storey as ref_storey,
                               coalesce(target.wall_position_index, 999999) as pos
                        {size_order}
                    """
                    result2 = self.engine.neo4j_conn.run(
                        cypher_no_storey,
                        subject_type=subject_type,
                        object_type=object_type,
                        object_material=object_material,
                        model=model_key,
                        **size_params,
                    )
                    candidates = [dict(r) for r in result2]

            # Final fallback: no topology edges at all → storey+type
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
            # 6.1.0: target_name_keyword is now rerank-only (descriptor, not filter).
            # GT-label vs IFC-name vocab overlap is 0/31 (audit 2026-04-28).
            return candidates

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
            # 6.1.0: target_name_keyword is now rerank-only (descriptor, not filter).
            return candidates

        # ── Phase 5B: new high-priority strategies (Neo4j graph queries) ──────
        elif strategy == "space+type":
            # IFCSpace-level query via Phase 1a method
            space_name = params.get("space_name", "")
            target_type = params.get("type", "")
            return self.engine.query_elements_in_space(space_name, ifc_type=target_type)

        # ── Original strategies ─────────────────────────────────────────────
        elif strategy == "storey+type":
            # Use IFCEngine's query_elements_by_level with type filter
            storey_name = params.get("storey", "")
            target_type = params.get("type", "")

            # Call v1 IFCEngine method
            results = self.engine.query_elements_by_level(storey_name)

            # Filter by type (with STARTS WITH for IFC subtypes,
            # e.g. IfcWall matches IfcWallStandardCase)
            def _type_match(r: dict) -> bool:
                rt = r.get("type") or ""
                ri = r.get("ifc_type") or ""
                return (rt == target_type or ri == target_type
                        or rt.startswith(target_type) or ri.startswith(target_type))
            return [r for r in results if _type_match(r)]

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

    def _execute_multi_anchor(
        self,
        subject_type: str,
        spatial_rels: List[Dict[str, Any]],
        storey_siblings: List[str],
        object_material: str,
        params: Dict[str, Any],
        *,
        include_relation_fingerprint: bool = True,
        include_position: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: Multi-anchor AND-intersection Cypher.

        All spatial_relations are hard filters (AND semantics).
        NEXT_TO relations additionally get a same-wall constraint via
        the wall_guid property on NEXT_TO edges, preventing false positives
        where two neighbors happen to be on different walls.

        Predicate routing:
          FILLS       → WHERE EXISTS clause (no edge props needed)
          NEXT_TO     → MATCH clause (accesses edge.wall_guid for same-wall)
          others      → WHERE EXISTS clause
        """
        if not self.engine.neo4j_conn:
            return []

        model_key = self.engine.model_key
        cypher_params: Dict[str, Any] = {
            "subject_type": subject_type,
            "model": model_key,
            "storey_list": storey_siblings,
        }

        fills_rels = [sr for sr in spatial_rels if sr.get("predicate") == "FILLS"]
        next_to_rels = [sr for sr in spatial_rels if sr.get("predicate") == "NEXT_TO"]
        other_rels = [sr for sr in spatial_rels
                      if sr.get("predicate") not in ("FILLS", "NEXT_TO", "CONTINUOUS")]

        match_lines = ["MATCH (target:IFCElement)"]
        where_parts = [
            "(target.ifc_type = $subject_type OR target.ifc_type STARTS WITH $subject_type)",
            "target.ifc_model = $model",
            "(size($storey_list) = 0"
            " OR ANY(s IN $storey_list WHERE toLower(target.storey) CONTAINS s))",
        ]

        # ── Phase 3B: FILLS+NEXT_TO wall-pinning ────────────────────────────
        # When FILLS and NEXT_TO co-occur, promote FILLS[0] from WHERE EXISTS
        # to a MATCH so we can tie nt_r{i}.wall_guid = fi0.guid.
        # Effect: pool drops from "all windows on storey NEXT_TO any door" (~43)
        # to "windows on the specific wall that also has a door" (~2-4).
        needs_exact_slot = include_position and params.get("position_index") is not None
        fills_as_match = bool(fills_rels and (next_to_rels or needs_exact_slot))

        # FILLS
        if fills_as_match:
            # Promote first FILLS rel to MATCH for wall-pinning
            sr0 = fills_rels[0]
            cypher_params["fills_obj_0"] = sr0.get("object_type", "")
            cypher_params["fills_mat_0"] = sr0.get("object_material") or ""
            match_lines.append("MATCH (target)-[:FILLS]->(fi0:IFCElement)")
            where_parts.append(
                "(fi0.ifc_type = $fills_obj_0 OR fi0.ifc_type STARTS WITH $fills_obj_0)"
            )
            where_parts.append("fi0.ifc_model = $model")
            where_parts.append(
                "($fills_mat_0 = '' OR toLower(fi0.material) CONTAINS toLower($fills_mat_0))"
            )
            # Fix 1: object_subtype on FILLS host (e.g. "brick" wall material filter)
            fills_subtype_0 = (sr0.get("object_subtype") or "").strip().lower()
            if fills_subtype_0:
                cypher_params["fills_subtype_0"] = fills_subtype_0
                where_parts.append(
                    "(toLower(coalesce(fi0.object_type, '')) CONTAINS $fills_subtype_0 "
                    "OR toLower(coalesce(fi0.name, '')) CONTAINS $fills_subtype_0)"
                )
            # Additional FILLS rels (rare) remain as WHERE EXISTS
            for i, sr in enumerate(fills_rels[1:], start=1):
                key = f"fills_obj_{i}"
                mat_key = f"fills_mat_{i}"
                cypher_params[key] = sr.get("object_type", "")
                cypher_params[mat_key] = sr.get("object_material") or ""
                fills_subtype_i = (sr.get("object_subtype") or "").strip().lower()
                subtype_clause_i = ""
                if fills_subtype_i:
                    cypher_params[f"fills_subtype_{i}"] = fills_subtype_i
                    subtype_clause_i = (
                        f" AND (toLower(coalesce(fi{i}.object_type, '')) CONTAINS $fills_subtype_{i} "
                        f"OR toLower(coalesce(fi{i}.name, '')) CONTAINS $fills_subtype_{i})"
                    )
                where_parts.append(
                    f"EXISTS {{ "
                    f"(target)-[:FILLS]->(fi{i}:IFCElement) "
                    f"WHERE (fi{i}.ifc_type = ${key} OR fi{i}.ifc_type STARTS WITH ${key}) "
                    f"AND fi{i}.ifc_model = $model "
                    f"AND (${mat_key} = '' OR toLower(fi{i}.material) CONTAINS toLower(${mat_key}))"
                    f"{subtype_clause_i}"
                    f" }}"
                )
        else:
            # FILLS only (no NEXT_TO): WHERE EXISTS (original behaviour)
            for i, sr in enumerate(fills_rels):
                key = f"fills_obj_{i}"
                mat_key = f"fills_mat_{i}"
                cypher_params[key] = sr.get("object_type", "")
                cypher_params[mat_key] = sr.get("object_material") or ""
                fills_subtype_i = (sr.get("object_subtype") or "").strip().lower()
                subtype_clause_i = ""
                if fills_subtype_i:
                    cypher_params[f"fills_subtype_{i}"] = fills_subtype_i
                    subtype_clause_i = (
                        f" AND (toLower(coalesce(fi{i}.object_type, '')) CONTAINS $fills_subtype_{i} "
                        f"OR toLower(coalesce(fi{i}.name, '')) CONTAINS $fills_subtype_{i})"
                    )
                where_parts.append(
                    f"EXISTS {{ "
                    f"(target)-[:FILLS]->(fi{i}:IFCElement) "
                    f"WHERE (fi{i}.ifc_type = ${key} OR fi{i}.ifc_type STARTS WITH ${key}) "
                    f"AND fi{i}.ifc_model = $model "
                    f"AND (${mat_key} = '' OR toLower(fi{i}.material) CONTAINS toLower(${mat_key}))"
                    f"{subtype_clause_i}"
                    f" }}"
                )

        # NEXT_TO: MATCH clause to access edge.wall_guid.
        # When fills_as_match, also pin to the specific FILLS wall (Phase 3B).
        for i, sr in enumerate(next_to_rels):
            key = f"nt_obj_{i}"
            cypher_params[key] = sr.get("object_type", "")
            match_lines.append(
                f"MATCH (target)-[nt_r{i}:NEXT_TO]->(nt_nb{i}:IFCElement)"
            )
            where_parts.append(
                f"(nt_nb{i}.ifc_type = ${key} OR nt_nb{i}.ifc_type STARTS WITH ${key})"
            )
            where_parts.append(f"nt_nb{i}.ifc_model = $model")
            # Phase 3B wall-pin: NEXT_TO edge must be on the same wall that target fills
            if fills_as_match:
                where_parts.append(f"nt_r{i}.wall_guid = fi0.guid")
            if include_relation_fingerprint:
                direction = sr.get("direction")
                if direction == "left":
                    where_parts.append(
                        f"coalesce(nt_nb{i}.wall_position_index, 999999) < "
                        f"coalesce(target.wall_position_index, -1)"
                    )
                elif direction == "right":
                    where_parts.append(
                        f"coalesce(nt_nb{i}.wall_position_index, -1) > "
                        f"coalesce(target.wall_position_index, 999999)"
                    )

                object_subtype = (sr.get("object_subtype") or "").strip().lower()
                if object_subtype:
                    subtype_key = f"nt_subtype_{i}"
                    cypher_params[subtype_key] = object_subtype
                    where_parts.append(
                        f"("
                        f"toLower(coalesce(nt_nb{i}.object_type, '')) CONTAINS ${subtype_key} "
                        f"OR toLower(coalesce(nt_nb{i}.name, '')) CONTAINS ${subtype_key}"
                        f")"
                    )

        # Same-wall constraint: all NEXT_TO edges must share the same wall
        if len(next_to_rels) >= 2:
            for i in range(len(next_to_rels) - 1):
                where_parts.append(f"nt_r{i}.wall_guid = nt_r{i + 1}.wall_guid")
            # All NEXT_TO neighbors must be distinct elements
            for i in range(len(next_to_rels)):
                for j in range(i + 1, len(next_to_rels)):
                    where_parts.append(f"nt_nb{i} <> nt_nb{j}")

        # Other predicates (ADJACENT_TO, CONNECTS_TO, etc.): WHERE EXISTS + material
        for i, sr in enumerate(other_rels):
            pred = sr.get("predicate", "")
            key = f"other_obj_{i}"
            mat_key = f"other_mat_{i}"
            cypher_params[key] = sr.get("object_type", "")
            cypher_params[mat_key] = sr.get("object_material") or ""
            # Fix 1: object_subtype on the anchor element (e.g. neighbour wall name)
            other_subtype = (sr.get("object_subtype") or "").strip().lower()
            subtype_clause = ""
            if other_subtype:
                subtype_key = f"other_subtype_{i}"
                cypher_params[subtype_key] = other_subtype
                subtype_clause = (
                    f" AND (toLower(coalesce(ot{i}.object_type, '')) CONTAINS ${subtype_key} "
                    f"OR toLower(coalesce(ot{i}.name, '')) CONTAINS ${subtype_key})"
                )
            # Fix 2: CONNECTS_TO connection_degree filter (COUNT sub-query on target node)
            degree_clause = ""
            if pred == "CONNECTS_TO":
                degree = sr.get("connection_degree")
                if degree is not None:
                    deg_key = f"other_degree_{i}"
                    cypher_params[deg_key] = int(degree)
                    degree_clause = (
                        f" AND COUNT {{ (target)-[:CONNECTS_TO]-() }} = ${deg_key}"
                    )
            # Fix 4: ADJACENT_TO distance_mm range filter on edge property (±200mm tolerance)
            dist_clause = ""
            if pred == "ADJACENT_TO":
                dist_mm = sr.get("distance_mm")
                if dist_mm is not None:
                    dist_key = f"other_dist_{i}"
                    cypher_params[dist_key] = float(dist_mm)
                    dist_clause = (
                        f" AND abs(coalesce(r_adj{i}.distance_mm, -9999) - ${dist_key}) <= 200"
                    )
            # Build the EXISTS / edge-var clause
            if pred == "ADJACENT_TO" and dist_clause:
                # Need to bind the relationship to access distance_mm
                where_parts.append(
                    f"EXISTS {{ "
                    f"(target)-[r_adj{i}:ADJACENT_TO]->(ot{i}:IFCElement) "
                    f"WHERE (ot{i}.ifc_type = ${key} OR ot{i}.ifc_type STARTS WITH ${key}) "
                    f"AND ot{i}.ifc_model = $model "
                    f"AND (${mat_key} = '' OR toLower(ot{i}.material) CONTAINS toLower(${mat_key}))"
                    f"{subtype_clause}"
                    f"{dist_clause}"
                    f" }}"
                    f"{degree_clause}"
                )
            else:
                where_parts.append(
                    f"EXISTS {{ "
                    f"(target)-[:{pred}]->(ot{i}:IFCElement) "
                    f"WHERE (ot{i}.ifc_type = ${key} OR ot{i}.ifc_type STARTS WITH ${key}) "
                    f"AND ot{i}.ifc_model = $model "
                    f"AND (${mat_key} = '' OR toLower(ot{i}.material) CONTAINS toLower(${mat_key}))"
                    f"{subtype_clause}"
                    f" }}"
                    f"{degree_clause}"
                )

        if needs_exact_slot:
            cypher_params["target_wall_position_index"] = max(int(params["position_index"]) - 1, 0)
            where_parts.append("coalesce(target.wall_position_index, -1) = $target_wall_position_index")
            if params.get("position_total") is not None:
                cypher_params["target_wall_child_total"] = int(params["position_total"])
                where_parts.append("coalesce(target.wall_child_total, -1) = $target_wall_child_total")

        # 6.1.1 / 6.1.6: size_cluster + size_band, mode-controlled.
        # ORDER BY must reference projected aliases under RETURN DISTINCT.
        # When both fields are present, size_band wins (broader, ResNet-derived).
        size_cluster_val = params.get("target_size_cluster")
        size_band_val = params.get("target_size_band")
        size_order = "ORDER BY pos"
        if size_band_val and self.size_band_mode != "off":
            cypher_params["target_size_band_prefix"] = f"{size_band_val}_"
            if self.size_band_mode == "hard":
                where_parts.append("target.size_cluster STARTS WITH $target_size_band_prefix")
            else:  # soft
                size_order = (
                    "ORDER BY CASE WHEN size_cluster STARTS WITH"
                    " $target_size_band_prefix THEN 0 ELSE 1 END, pos"
                )
        elif size_cluster_val and self.size_cluster_mode != "off":
            cypher_params["target_size_cluster"] = str(size_cluster_val)
            if self.size_cluster_mode == "hard":
                where_parts.append("target.size_cluster = $target_size_cluster")
            else:  # soft
                size_order = (
                    "ORDER BY CASE WHEN size_cluster = $target_size_cluster"
                    " THEN 0 ELSE 1 END, pos"
                )

        match_str = "\n    ".join(match_lines)
        where_str = "\n      AND ".join(where_parts)
        cypher = f"""
        {match_str}
        WHERE {where_str}
        RETURN DISTINCT target.guid AS guid, target.name AS name,
               target.ifc_type AS type,
               target.storey AS storey,
               target.object_type AS object_type,
               target.size_cluster AS size_cluster,
               target.wall_child_total AS wall_child_total,
               target.wall_position_index AS pos
        {size_order}
        """
        try:
            return [dict(r) for r in self.engine.neo4j_conn.run(cypher, **cypher_params)]
        except Exception:
            return []

    def _relax_multi_anchor(
        self,
        subject_type: str,
        spatial_rels: List[Dict[str, Any]],
        storey_siblings: List[str],
        object_material: str,
        params: Dict[str, Any],
        *,
        include_relation_fingerprint: bool = True,
        include_position: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Relaxation ladder for multi-anchor: drop the lowest-confidence SR
        one at a time until candidates are found.

        Falls back to single-hop with the highest-confidence SR if all
        subsets of size >= 2 return empty.
        """
        # Sort ascending by confidence — drop weakest first
        ranked = sorted(spatial_rels, key=lambda x: x.get("confidence", 1.0))
        remaining = list(ranked)

        while len(remaining) > 1:
            remaining = remaining[1:]  # drop weakest
            if len(remaining) == 1:
                break
            candidates = self._execute_multi_anchor(
                subject_type,
                remaining,
                storey_siblings,
                object_material,
                params,
                include_relation_fingerprint=include_relation_fingerprint,
                include_position=include_position,
            )
            if candidates:
                return candidates
            # Also try without storey
            if storey_siblings:
                candidates = self._execute_multi_anchor(
                    subject_type,
                    remaining,
                    [],
                    object_material,
                    params,
                    include_relation_fingerprint=include_relation_fingerprint,
                    include_position=include_position,
                )
                if candidates:
                    return candidates

        # Single-SR fallback: use highest-confidence relation
        best = max(spatial_rels, key=lambda x: x.get("confidence", 1.0))
        return self._execute_multi_anchor(
            subject_type,
            [best],
            storey_siblings,
            object_material,
            params,
            include_relation_fingerprint=include_relation_fingerprint,
            include_position=include_position,
        )

    def _has_relation_fingerprint(
        self,
        spatial_rels: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> bool:
        if params.get("position_index") is not None:
            return True
        return any((sr.get("direction") or sr.get("object_subtype")) for sr in spatial_rels)

    def _requires_multi_chain(
        self,
        spatial_rels: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> bool:
        if len(spatial_rels) >= 2:
            return True
        if self._has_relation_fingerprint(spatial_rels, params):
            return True
        return False

    def _get_storey_type_pool(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run a storey+type query using the same params as the P0 plan.
        Used by the P0 ∩ P1 defensive intersection.
        Returns [] if storey or subject_type is missing.

        6.1.6: when `target_size_band` (hard mode) or `target_size_cluster`
        (hard mode) is in scope, filter P1 by the same constraint via Cypher.
        Otherwise unfiltered P1 swamps band-filtered P0 in the p0_union_p1
        path, washing out the band signal.
        """
        storey = params.get("storey", "")
        subject_type = params.get("subject_type", "")
        if not storey or not subject_type:
            return []

        # Direct Cypher path when neo4j is up — needed to apply band/cluster
        # filters that the legacy storey+type executor doesn't carry.
        if self.retrieval_mode == "neo4j" and getattr(self.engine, "neo4j_conn", None):
            storey_siblings = self._resolve_storey_siblings(storey)
            cypher_params: Dict[str, Any] = {
                "type": subject_type,
                "storey_list": storey_siblings,
                "model": getattr(self.engine, "model_key", "AP"),
            }
            size_where = ""
            band_val = params.get("target_size_band")
            cluster_val = params.get("target_size_cluster")
            if band_val and self.size_band_mode == "hard":
                cypher_params["target_size_band_prefix"] = f"{band_val}_"
                size_where = "\n              AND e.size_cluster STARTS WITH $target_size_band_prefix"
            elif cluster_val and self.size_cluster_mode == "hard":
                cypher_params["target_size_cluster"] = str(cluster_val)
                size_where = "\n              AND e.size_cluster = $target_size_cluster"
            cypher = f"""
                MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
                WHERE (e.ifc_type = $type OR e.ifc_type STARTS WITH $type)
                  AND e.ifc_model = $model
                  AND (size($storey_list) = 0
                       OR ANY(alias IN $storey_list WHERE toLower(s.name) CONTAINS alias)){size_where}
                RETURN e.guid AS guid, e.name AS name, e.ifc_type AS type,
                       e.size_cluster AS size_cluster, s.name AS storey,
                       coalesce(e.wall_position_index, 999999) AS pos
                ORDER BY pos
            """
            try:
                return [dict(r) for r in self.engine.neo4j_conn.run(cypher, **cypher_params)]
            except Exception:
                pass  # fall through to legacy path

        # Memory-mode / no-neo4j fallback (no band filter — kept for back-compat)
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
