import ifcopenshell
import ifcopenshell.geom
import open3d as o3d

# 1. 打开文件
model = ifcopenshell.open("example.ifc")

# 2. 设置几何转换设置 (转为三角形 Mesh)
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

# 3. 提取墙体并转换
vis_elements = []
for wall in model.by_type("IfcWall"):
    try:
        # 创建几何形状对象
        shape = ifcopenshell.geom.create_shape(settings, wall)
        
        # 提取顶点和面 (这是最麻烦的一步，需要从 shape.geometry 提取 raw data)
        # 这里只是伪代码逻辑，实际需要处理顶点索引
        verts = shape.geometry.verts 
        faces = shape.geometry.faces 
        
        # 4. 放入 Open3D
        mesh = o3d.geometry.TriangleMesh()
        # ... 填充数据到 mesh ...
        vis_elements.append(mesh)
    except:
        pass

# 5. 弹窗显示
o3d.visualization.draw_geometries(vis_elements)