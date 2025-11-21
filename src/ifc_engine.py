import ifcopenshell
import ifcopenshell.util.element
import os

class IFCEngine:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"IFC file not found: {file_path}")
        
        print(f"🏗️  Loading IFC Model: {os.path.basename(file_path)}...")
        self.file = ifcopenshell.open(file_path)
        self.spatial_index = {} 
        self._build_spatial_graph()

    def _build_spatial_graph(self):
        """
        构建空间拓扑索引 (The Topological Semantics Layer).
        解析 IfcSpace -> IfcRelContainedInSpatialStructure -> Elements
        """
        print("⚙️  Building Semantic Graph Index...")
        spaces = self.file.by_type("IfcSpace")
        
        for space in spaces:
            # 获取房间名 (优先取 LongName，其次 Name)
            room_name = space.LongName if space.LongName else space.Name
            if not room_name:
                continue
            
            # 归一化为小写以便查询
            key = room_name.lower()
            self.spatial_index[key] = []

            # 核心：利用 util.element 获取空间内的构件
            elements = ifcopenshell.util.element.get_decomposition(space)
            
            for el in elements:
                # 过滤掉不需要的 Opening (如门窗洞口)
                if el.is_a("IfcOpeningElement") or el.is_a("IfcSpace"): 
                    continue
                    
                self.spatial_index[key].append({
                    "guid": el.GlobalId,
                    "type": el.is_a(), # 例如 IfcWall, IfcSlab
                    "name": el.Name if el.Name else "Unnamed",
                    "description": el.Description if hasattr(el, "Description") else ""
                })
        
        print(f"✅  Graph Index Ready: {len(self.spatial_index)} spaces indexed.")

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

    def get_element_properties(self, guid: str):
        """
        获取属性集 (Mock Compliance Check)
        """
        try:
            element = self.file.by_id(guid)
            # 简单返回，实际可扩展为读取 Pset_WallCommon 等
            return str({
                "GlobalId": element.GlobalId,
                "Name": element.Name,
                "Type": element.is_a(),
                "PredefinedType": element.ObjectType if hasattr(element, "ObjectType") else "N/A"
            })
        except:
            return "Element not found."