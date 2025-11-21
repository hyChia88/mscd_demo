# src/inspect_ifc.py
import sys
import os

# 确保能找到模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ifc_engine import IFCEngine

# 路径配置 (根据你终端里的路径调整)
IFC_PATH = "data/Building-Architecture.ifc"

def inspect():
    if not os.path.exists(IFC_PATH):
        print(f"❌ 找不到文件: {IFC_PATH}")
        return

    print(f"🔍 正在分析: {IFC_PATH} ...")
    engine = IFCEngine(IFC_PATH)
    
    print("\n" + "="*30)
    print(f"📊 统计: 发现 {len(engine.spatial_index)} 个空间 (Rooms/Spaces)")
    print("="*30)
    
    for room_name, elements in engine.spatial_index.items():
        print(f"\n🏠 房间名 (Key): '{room_name}'")
        print(f"   └── 包含 {len(elements)} 个构件")
        
        # 打印前 5 个构件看看是什么
        for i, el in enumerate(elements[:5]):
            print(f"       - [{el['type']}] {el['name']} (GUID: {el['guid']})")
        
        if len(elements) > 5:
            print(f"       - ... 还有 {len(elements)-5} 个")

if __name__ == "__main__":
    inspect()