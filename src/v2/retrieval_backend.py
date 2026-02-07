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
        use_clip: bool = False
    ):
        """
        Initialize retrieval backend.

        Args:
            engine: IFCEngine instance from v1
            retrieval_mode: "memory" or "neo4j"
            visual_aligner: VisualAligner instance (optional)
            use_clip: Whether to enable CLIP reranking
        """
        self.engine = engine
        self.retrieval_mode = retrieval_mode
        self.visual_aligner = visual_aligner
        self.use_clip = use_clip and (visual_aligner is not None)

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
        # Step 1: Execute base query
        if self.retrieval_mode == "neo4j":
            candidates = self._execute_neo4j(plan)
        else:
            candidates = self._execute_memory(plan)

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
            rerank_applied=rerank_applied
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

        if strategy == "storey+type":
            # Most specific: filter by storey AND type
            storey_key = params.get("storey", "").lower()
            target_type = params.get("type", "")

            results = self.engine.find_elements_in_space(storey_key)
            return [r for r in results if r.get("type") == target_type]

        elif strategy == "storey_only":
            # Filter by storey only
            storey_key = params.get("storey", "").lower()
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
                    name = element.get("name", "").lower()
                    desc = element.get("description", "").lower()

                    # Check if any keyword matches
                    for keyword in keywords:
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

        if strategy == "storey+type":
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
