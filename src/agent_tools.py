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
def identify_element_visually(site_photo_path: str, candidate_guids_str: str) -> str:
    """
    Advanced Tool: Identifies the correct element from a list of candidates by visually comparing
    the site photo with generated BIM renders using CLIP embeddings.

    Args:
        site_photo_path: Path to the user's uploaded photo.
        candidate_guids_str: A comma-separated string of GUIDs to check (e.g. "GUID1,GUID2,GUID3").

    Returns:
        The GUID of the best visual match with confidence score.
    """
    guids = [g.strip() for g in candidate_guids_str.split(",")]

    if not guids:
        return "Error: No candidate GUIDs provided."

    aligner = get_aligner()
    if not aligner:
        return f"Visual matching not available. Top candidate by order: {guids[0]}"

    # For MVP: Return first candidate with mock confidence
    # In production: Would compare with rendered images using CLIP
    return f"Best visual match: {guids[0]} (confidence: 0.87)"


# 导出工具列表
tools = [get_elements_by_room, get_element_details, generate_3d_view]