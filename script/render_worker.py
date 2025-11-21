import bpy
import sys
import os
import math
from mathutils import Vector

# --- 参数解析 ---
argv = sys.argv
try:
    index = argv.index("--") + 1
except ValueError:
    print("Error: No arguments passed after '--'")
    sys.exit(1)

ifc_path = argv[index]
output_path = argv[index+1]
target_guid = argv[index+2]

# --- 1. 初始化环境 ---
# 启用 BlenderBIM (确保已安装)
if not "bonsai" in bpy.context.preferences.addons and "blenderbim" not in bpy.context.preferences.addons:
    print("Error: Bonsai/BlenderBIM addon not found!")
    sys.exit(1)

# 清空场景
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 加载 IFC
print(f"Loading {ifc_path}...")
bpy.ops.bim.load_project(filepath=ifc_path)

# --- 2. 查找目标并计算包围盒 ---
target_obj = None

# 在 BlenderBIM 中，导入的物体通常以 IfcType/Name 命名，属性在 data 中
# 这是一个简化的查找逻辑，遍历所有物体检查 GlobalId
for obj in bpy.context.scene.objects:
    # Bonsai 属性通常存储在 obj.BIMObjectProperties.ifc_definition_id
    # 但为了通用性，我们尝试从属性或名称匹配
    if hasattr(obj, "BIMObjectProperties"):
        if obj.BIMObjectProperties.ifc_definition_id == target_guid:
            target_obj = obj
            break

if not target_obj:
    print(f"Warning: GUID {target_guid} not found in 3D view.")
    # 备用：如果找不到，就不渲染或渲染全局
    sys.exit(0)

# --- 3. 相机自动对焦逻辑 (Auto-Focus Logic) ---
# 选中目标
bpy.ops.object.select_all(action='DESELECT')
target_obj.select_set(True)
bpy.context.view_layer.objects.active = target_obj

# 计算中心点
bbox_corners = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
center = sum(bbox_corners, Vector()) / 8
size = (bbox_corners[6] - bbox_corners[0]).length # 对角线长度

# 创建相机
cam_data = bpy.data.cameras.new(name='Camera')
cam_obj = bpy.data.objects.new(name='Camera', object_data=cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# 设置相机位置：在物体前方 +Y 方向，高度稍微抬高
# 距离取决于物体大小 (size)
dist = size * 2.0
if dist < 2.0: dist = 2.0 # 最小距离

cam_pos = center + Vector((0, -dist, dist * 0.5)) # 简单的 ISO 视角
cam_obj.location = cam_pos

# 让相机看向物体中心 (Look At)
direction = center - cam_obj.location
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()

# --- 4. 渲染设置 ---
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 800
scene.render.resolution_y = 600
scene.render.filepath = output_path

# 隐藏不需要的构件 (可选：只显示目标)
# for obj in scene.objects:
#    if obj != target_obj and obj != cam_obj:
#        obj.hide_render = True

print(f"Rendering to {output_path}...")
bpy.ops.render.render(write_still=True)