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

    def __init__(self, file_path: str, neo4j_conn=None):
        """
        Initialize IFC Engine with optional Neo4j connection.

        Args:
            file_path: Path to IFC file
            neo4j_conn: Optional py2neo Graph connection for graph export
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"IFC file not found: {file_path}")

        print(f"🏗️  Loading IFC Model: {os.path.basename(file_path)}...")
        self.file = ifcopenshell.open(file_path)
        self.file_path = file_path
        self.spatial_index = {}
        self.neo4j_conn = neo4j_conn
        self._build_spatial_graph()

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

        print(f"✅  Graph Index Ready: {len(self.spatial_index)} groups indexed.")

    def find_elements_in_space(self, room_query: str):
        """
        根据房间名模糊查找 (Semantic Search Simulation)
        """
        room_query = room_query.lower()
        found_elements = []
        
        # 简单的包含匹配 (在完整 Thesis 中这里可以是 Vector Search)
        for room_name, elements in self.spatial_index.items():
            if room_query in room_name:
                found_elements.extend(elements)
        
        return found_elements

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
                "elevation": getattr(storey, "Elevation", 0)
            })
            count += 1

        # Spaces
        for space in self.file.by_type("IfcSpace"):
            self._create_node("IFCSpace", {
                "guid": space.GlobalId,
                "name": space.Name or space.LongName,
                "long_name": space.LongName
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
            "IfcFurnishingElement",  # IFC2X3 compatible
            "IfcFurniture",  # IFC4 only
        ]

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
                }

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

    def _create_spatial_relationships(self) -> int:
        """Create CONTAINS_IN relationships between spaces and elements"""
        count = 0

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
                count += 1

        return count

    def _create_element_relationships(self) -> int:
        """Create relationships between elements (voids, fills, connections)"""
        count = 0

        # IfcRelVoidsElement: Opening relationships
        for rel in self.file.by_type("IfcRelVoidsElement"):
            self._create_relationship(
                "IFCElement",
                rel.RelatingBuildingElement.GlobalId,
                "HAS_OPENING",
                "IFCElement",
                rel.RelatedOpeningElement.GlobalId
            )
            count += 1

        # IfcRelFillsElement: Door/Window fills opening
        for rel in self.file.by_type("IfcRelFillsElement"):
            self._create_relationship(
                "IFCElement",
                rel.RelatedBuildingElement.GlobalId,
                "FILLS",
                "IFCElement",
                rel.RelatingOpeningElement.GlobalId
            )
            count += 1

        return count

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

    # =========================================================================
    # Graph Query Methods (for Agent Reasoning)
    # =========================================================================

    def query_elements_by_level(self, level_name: str) -> List[Dict]:
        """
        Query elements on a specific level/storey via Neo4j.

        Useful for RQ3 abductive reasoning: "Which elements are on Level 6?"
        """
        if not self.neo4j_conn:
            # Fallback to local spatial index
            return self.find_elements_in_space(level_name)

        query = """
        MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
        WHERE toLower(s.name) CONTAINS toLower($level_name)
        RETURN e.guid as guid, e.name as name, e.ifc_type as type,
               e.FireRating as fire_rating, e.LoadBearing as load_bearing
        """
        result = self.neo4j_conn.run(query, level_name=level_name)
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
        RETURN e.guid as guid, e.name as name, e.ifc_type as type
        """
        result = self.neo4j_conn.run(query, value=property_value)
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
        RETURN DISTINCT related.guid as guid, related.name as name, related.ifc_type as type
        """
        result = self.neo4j_conn.run(query, guid=guid)
        return [dict(record) for record in result]