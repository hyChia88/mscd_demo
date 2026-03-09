Neo4j 在这个项目中的工作原理
什么是 Neo4j？
Neo4j 是一个图数据库（Graph Database），与传统的表格数据库不同，它用**节点（Node）和关系（Relationship）**来存储数据。


传统数据库（表格）:
┌─────────┬──────────┬─────────────┐
│ 元素ID   │ 类型      │ 楼层         │
├─────────┼──────────┼─────────────┤
│ 001     │ IfcWindow │ 6th Floor   │
│ 002     │ IfcWall   │ 6th Floor   │
└─────────┴──────────┴─────────────┘

图数据库（Neo4j）:
    [6th Floor]
        │
   CONTAINS（包含）
        │
        ▼
    [Window 001] ──FILLS（填充）──▶ [Opening] ──IN_WALL──▶ [Wall 002]
为什么 BIM 适合用图数据库？
IFC/BIM 模型本身就是语义关系网络：


建筑 ──HAS_STOREY──▶ 楼层 ──CONTAINS──▶ 房间 ──BOUNDED_BY──▶ 墙
                                              │
                                         HAS_OPENING
                                              │
                                              ▼
                                            开口 ◀──FILLS── 窗户
两种查询模式对比
特性	Memory 模式	Neo4j 模式
数据结构	Python 字典（空间索引）	图数据库
查询方式	按楼层/房间查找元素	图遍历（关系查询）
查询示例	"6楼有哪些窗户？"	"这扇窗户连接到哪些墙？"
启动速度	快	需要连接数据库
适用场景	简单空间查询	复杂关系推理
项目中的工作流程

┌─────────────────────────────────────────────────────────────────┐
│                        工作流程                                  │
└─────────────────────────────────────────────────────────────────┘

1️⃣ 数据准备（一次性）
   ┌──────────────┐      python script/ifc_to_neo4j.py
   │ IFC 文件      │  ─────────────────────────────────▶  [Neo4j 数据库]
   │ (.ifc)       │         导出节点和关系
   └──────────────┘

2️⃣ 运行实验
   
   Memory 模式:
   $ python src/main_mcp.py --experiment memory
   
   ┌─────────────┐     MCP协议      ┌─────────────────┐
   │ LLM Agent   │ ◀──────────────▶ │ ifc_server.py   │
   │ (Gemini)    │                  │ ┌─────────────┐ │
   └─────────────┘                  │ │ 空间索引     │ │  ← 直接从 Python 字典查询
                                    │ │ (dict)      │ │
                                    │ └─────────────┘ │
                                    └─────────────────┘
   
   Neo4j 模式:
   $ python src/main_mcp.py --experiment neo4j
   
   ┌─────────────┐     MCP协议      ┌─────────────────┐     Cypher查询
   │ LLM Agent   │ ◀──────────────▶ │ ifc_server.py   │ ◀─────────────▶ [Neo4j]
   │ (Gemini)    │                  │                 │
   └─────────────┘                  └─────────────────┘
具体代码流程
第一步：导出 IFC 到 Neo4j


# 启动 Neo4j 容器
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 neo4j:latest

# 导出 IFC 模型到图数据库
python script/ifc_to_neo4j.py
这会创建类似这样的图结构：


(:IFCStorey {name: "6 - Sixth Floor"})
    │
    ├──[:CONTAINS]──▶ (:IFCWindow {guid: "0Um_J2ClP45...", name: "Window_001"})
    │
    └──[:CONTAINS]──▶ (:IFCWall {guid: "1KMtYLy...", FireRating: "REI120"})
第二步：运行对比实验


# 实验1：内存模式（基线）
python src/main_mcp.py --experiment memory
# 输出: logs/evaluations/eval_20260127_120000_memory.json

# 实验2：Neo4j 模式
python src/main_mcp.py --experiment neo4j
# 输出: logs/evaluations/eval_20260127_120000_neo4j.json
Neo4j 模式的优势
示例查询：找到窗户相邻的元素

Memory 模式只能返回：


{"same_space": "6 - Sixth Floor", "nearby_elements": [所有6楼元素]}
Neo4j 模式可以返回语义关系：


{
  "adjacent_elements": [
    {"type": "IfcWall", "relationship": "HAS_OPENING"},
    {"type": "IfcCurtainWall", "relationship": "BOUNDED_BY"}
  ]
}
配置文件说明
config.yaml 中的设置：


neo4j:
  enabled: false      # 默认关闭，使用 memory 模式
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "password"
优先级：--experiment 参数 > 环境变量 QUERY_MODE > config.yaml

什么时候用哪种模式？
场景	推荐模式
快速测试/开发	memory
简单查询（按楼层找元素）	memory
复杂关系查询（找相邻元素）	neo4j
合规性检查（属性传播）	neo4j
论文实验对比	两种都跑
