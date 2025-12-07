import os
import yaml
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

def load_system_prompt(prompt_file="prompts/system_prompt.yaml"):
    """Load system prompt from YAML file"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, prompt_file)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("system_prompt", "")

"""
Purpose: Demonstrate how an AI Agent processes building inspection reports

5 Key Steps:

1. Initialize LLM (lines 27-31)
- Use Google Gemini 2.5 Flash model
- Temperature set to 0 (deterministic responses)

2. Define Agent Persona (lines 33-66)
- Tell Agent: You are an "Interpreter Layer"
- You have these tools: get_elements_by_room, get_element_details, generate_3d_view
- This is your workflow: Extract location from report → Query IFC → Filter results → Extract details → Output report

3. Create Agent (lines 72-75)
- Assemble Agent with: LLM + Tools + System Prompt
- This way Agent knows what to do

4. Load Test Scenarios (lines 77-78)
- Read 2 real-world scenarios from test.yaml

5. Execute Scenarios in Loop (line 80+)
- For each scenario:
    - Print input question
    - Agent thinks + calls tools
    - Print Agent's final answer
    - Wait 3 seconds, move to next scenario

Simple Flow Diagram:

Start main.py
↓
Initialize Gemini LLM
↓
Create Agent (give it tools + system prompt)
↓
Read test.yaml (2 scenarios)
↓
[Loop] For each scenario:
Scenario 1 → Agent thinks → Calls tools → Outputs result
Scenario 2 → Agent thinks → Calls tools → Outputs result
↓
Done!
"""

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

    # 2. Load System Prompt from YAML
    try:
        system_prompt = load_system_prompt()
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        print("Using fallback system prompt...")
        system_prompt = "You are a helpful BIM inspection assistant. Use the available tools to help users query the IFC model."

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