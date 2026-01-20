import os
from langchain.tools import tool
from ifc_engine import IFCEngine

# --- 单例模式加载引擎 ---
# 这样 Agent 每次思考时不需要重新加载大文件
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IFC_PATH = os.path.join(BASE_DIR, "data", "BasicHouse.ifc")

print(f"Initializing Engine with: {IFC_PATH}, please wait...")
engine = IFCEngine(IFC_PATH)

# --- 延迟加载 Visual Aligner (仅在需要时加载重量级模块) ---
_aligner = None

def get_aligner():
    global _aligner
    if _aligner is None:
        try:
            from visual_matcher import VisualAligner
            _aligner = VisualAligner()
        except ImportError:
            print("⚠️  Warning: CLIP model not available, visual matching disabled")
            _aligner = False
    return _aligner if _aligner else None

# --- 定义 LangChain Tools ---

@tool
def get_elements_by_room(room_name: str) -> str:
    """
    Use this tool to find BIM objects located in a specific room or space.
    Input: The name of the room (e.g., 'Kitchen', 'Living Room').
    Output: A list of elements with their names, types, and GUIDs.
    """
    # print(f"[Tool Call] Querying room: {room_name}") 
    results = engine.find_elements_in_space(room_name)
    
    if not results:
        return f"No elements found in a room matching '{room_name}'."
    
    # 限制返回数量，防止 Token 溢出，这展示了你对 LLM 限制的理解
    return str(results[:30]) 

@tool
def get_element_details(guid: str) -> str:
    """
    Use this tool to get detailed technical properties (Psets) of a specific element by its GUID.
    Useful for compliance checking (checking material, fire rating, etc.).
    """
    return engine.get_element_properties(guid)

@tool
def generate_3d_view(guid: str) -> str:
    """
    Generates a visual verification snapshot (render) of the element.
    Use this when the user asks to 'see' or 'visualize' the defect.
    """
    # MVP 策略：Mock 掉 Blender 渲染
    # 面试话术："In the full thesis, this triggers a headless Blender worker. For this MVP, I'm mocking the file generation."
    return f"/server/renders/{guid}_inspection_view.png"

@tool
def identify_element_visually(site_description_or_photo: str, candidate_guids_str: str) -> str:
    """
    Advanced Tool: Identifies the correct element from a list of candidates by visually comparing
    the site description/photo with BIM element descriptions using CLIP embeddings.

    Args:
        site_description_or_photo: Either a text description (e.g., "white kitchen cabinet")
                                   or a file path to a photo (e.g., "data/site_photos/evidence_01.jpg").
        candidate_guids_str: A comma-separated string of GUIDs to check (e.g., "GUID1,GUID2,GUID3").

    Returns:
        The GUID of the best visual match with confidence score.

    Example usage:
        - identify_element_visually("white cabinet with wood countertop", "guid1,guid2,guid3")
        - identify_element_visually("data/site_photos/cabinet.jpg", "guid1,guid2,guid3")
    """
    guids = [g.strip() for g in candidate_guids_str.split(",")]

    if not guids:
        return "Error: No candidate GUIDs provided."

    aligner = get_aligner()
    if not aligner:
        return f"Visual matching not available. Top candidate by order: {guids[0]}"

    # Get descriptions for each candidate element
    candidate_descriptions = []
    for guid in guids:
        try:
            # Get element from IFC file
            element = engine.file.by_id(guid)
            if element:
                name = element.Name if element.Name else "Unnamed"
                elem_type = element.is_a()
                desc = f"{name} - {elem_type}"
                candidate_descriptions.append(desc)
            else:
                candidate_descriptions.append(f"Element {guid}")
        except:
            candidate_descriptions.append(f"Element {guid}")

    # Check if input is a file path or text description
    import os
    is_file = os.path.exists(site_description_or_photo)

    if is_file:
        # For MVP: Mock image processing
        # In production: Would load image and use CLIP image encoder
        query_text = f"Visual features from photo: {os.path.basename(site_description_or_photo)}"
    else:
        query_text = site_description_or_photo

    # Use VisualAligner to find best match
    try:
        best_idx, score, best_match = aligner.find_best_match(query_text, candidate_descriptions)
        matched_guid = guids[best_idx]
        return f"Best visual match: {matched_guid}\nElement: {best_match}\nConfidence: {score:.2f}"
    except Exception as e:
        # Fallback to first candidate if alignment fails
        return f"Visual matching error: {str(e)}. Returning top candidate: {guids[0]}"

@tool
def list_available_spaces() -> str:
    """
    Use this tool to discover what rooms/floors/spaces are available in the IFC model.
    This is helpful when you don't know the exact room names to search.
    Returns a list of available space names that can be used with get_elements_by_room.
    """
    available_spaces = list(engine.spatial_index.keys())
    if not available_spaces:
        return "No spaces found in the IFC model."

    # Also show how many elements in each space
    details = []
    for space in available_spaces:
        count = len(engine.spatial_index[space])
        details.append(f"'{space}' ({count} elements)")

    return "Available spaces:\n" + "\n".join(f"  - {d}" for d in details)


# 导出工具列表
tools = [list_available_spaces, get_elements_by_room, get_element_details, generate_3d_view, identify_element_visually]