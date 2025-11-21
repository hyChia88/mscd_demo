import os
from langchain.tools import tool
from ifc_engine import IFCEngine
from visual_matcher import VisualAligner
from blender_service import run_blender_render

# --- 单例模式加载引擎 ---
# 这样 Agent 每次思考时不需要重新加载大文件
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IFC_PATH = os.path.join(BASE_DIR, "data", "Building-Architecture.ifc")

print(f"Initializing Engine with: {IFC_PATH}, please wait...")
engine = IFCEngine(IFC_PATH)
aligner = VisualAligner()

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
def identify_element_visually(site_photo_path: str, candidate_guids_str: str):
    """
    Advanced Tool: Identifies the correct element from a list of candidates by visually comparing 
    the site photo with generated BIM renders.
    
    Args:
        site_photo_path: Path to the user's uploaded photo.
        candidate_guids_str: A comma-separated string of GUIDs to check (e.g. "GUID1,GUID2,GUID3").
        
    Returns:
        The GUID of the best visual match.
    """
    guids = candidate_guids_str.split(",")
    candidate_renders = {}
    
    # 1. 批量渲染 (Batch Rendering) - 这里调用 Headless Blender
    # 这一步对应论文 T1 任务：从 Site Evidence 到 IFC Elements 的链接 
    for guid in guids:
        guid = guid.strip()
        # 假设 run_blender_render 会返回生成的图片路径
        render_path = run_blender_render(guid) 
        candidate_renders[guid] = render_path
    
    # 2. 视觉排序 (Visual Ranking) - 这里对应 RQ1 的 Top-1 Retrieval 
    ranked = aligner.rank_candidates(site_photo_path, candidate_renders)
    
    if not ranked:
        return "Error: Visual alignment failed."
        
    best_guid, best_score = ranked[0]
    
    # 3. 设置阈值 (Calibration)
    if best_score < 0.65: # 阈值需要实验调试
        return f"Uncertain result. Best match was {best_guid} but score ({best_score:.2f}) is too low."
        
    return f"Visual Match Found: {best_guid} (Confidence: {best_score:.2f})"

# 导出工具列表
tools = [get_elements_by_room, get_element_details, generate_3d_view]