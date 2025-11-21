import os
import yaml
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

from agent_tools import tools

def load_scenarios(file_path="test.yaml"):
    """读取测试剧本"""
    if not os.path.exists(file_path):
        print(f"❌ Error: Configuration file '{file_path}' not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML: {e}")
            return []

def main():
    # 1. 设置 LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
    )

    # 2. 设计 Agent Persona
    system_prompt = """
    You are an advanced AI "Interpreter Layer" for AEC projects.
    Your mission is to bridge unstructured site reports with the structured IFC building model.

    YOU HAVE ACCESS TO:
    - IFC database organized by Floors: "Floor 0" (residential level with 123 elements), "Floor 1" (roof level)
    - Tools:
      1. get_elements_by_room(floor_name) - Returns ALL elements on that floor
      2. get_element_details(guid) - Returns properties like Type, Name, ObjectType for compliance checks
      3. generate_3d_view(guid) - Returns a render path for visual verification

    CRITICAL WORKFLOW FOR EVERY USER REPORT:
    1. **Always Start Here**: Extract the floor name from the report (e.g., "Floor 0", "Floor 1")
    2. **Query the Floor**: Call get_elements_by_room with the floor name
    3. **Semantic Filtering**: Search the results for:
       - Element names containing keywords (e.g., "Cabinet", "Wall", "Door", "Yttervägg", "Innerdörr")
       - Element types (IfcWall, IfcDoor, IfcWindow, IfcFurniture, IfcSlab, etc.)
    4. **Compliance Check**: Call get_element_details(guid) for each identified element
    5. **Comprehensive Report**: List all found elements with:
       - Element Name
       - Element Type
       - GUID
       - Key Properties (Material, Fire Rating, etc. from get_element_details)

    KEY FACTS ABOUT THIS MODEL:
    - Floor 0 contains: walls (outer & inner), doors, windows, furniture (cabinets, beds, tables, etc.)
    - Floor 1 contains: roof structure, slabs
    - Element names include Swedish names like "Yttervägg" (exterior wall), "Innerdörr" (interior door)
    - Furniture items start with "M_" (e.g., "M_Base Cabinet", "M_Bed")

    ALWAYS FOLLOW THIS PATTERN:
    User mentions "Floor 0" → Call get_elements_by_room("floor 0") → Search results → Call get_element_details for matches → Report
    """

    # 3. 组装 Agent (使用 LangGraph ReAct agent)
    # Prepend system prompt to the LLM
    from langchain_core.messages import SystemMessage

    agent_executor = create_react_agent(
        llm.bind(system=system_prompt),
        tools
    )

    # 4. 加载并执行测试剧本
    scenarios = load_scenarios()

    print(f"\n✅ === Agent initialized. Loaded {len(scenarios)} scenarios. ===\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"📋 Scenario {i}: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Description: {scenario['description']}\n")

        user_input = scenario['input']
        print(f"📥 Input:\n{user_input}\n")

        try:
            # 记录开始时间
            start_time = time.time()

            response = agent_executor.invoke({"messages": [("user", user_input)]})

            elapsed = time.time() - start_time

            print(f"\n📤 Final Response ({elapsed:.2f}s):")
            print("-" * 70)
            # Extract the response from the message
            if "messages" in response:
                output = response["messages"][-1].content
            else:
                output = str(response)
            print(output)
            print("-" * 70)

        except Exception as e:
            print(f"\n❌ Error during execution: {e}")

        # 演示时的停顿
        if i < len(scenarios):
            print(f"\n⏳ ...Proceeding to next scenario in 3 seconds...")
            time.sleep(3)

if __name__ == "__main__":
    main()