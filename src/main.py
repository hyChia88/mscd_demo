import os
import yaml  # 新增：用于读取 yaml
import time  # 新增：用于演示时的停顿
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from colorama import Fore, Style, init

# 初始化颜色输出
init(autoreset=True)
load_dotenv()

from agent_tools import tools

def load_scenarios(file_path="test.yaml"):
    """读取测试剧本"""
    if not os.path.exists(file_path):
        print(f"{Fore.RED}Error: Configuration file '{file_path}' not found.{Style.RESET_ALL}")
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"{Fore.RED}Error parsing YAML: {e}{Style.RESET_ALL}")
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
    You are an advanced AI "Interpreter Layer" for AEC.
    Your mission is to bridge unstructured site data with the structured IFC Graph.

    WORKFLOW:
    1. **Contextualize**: Identify the location (Room/Space).
    2. **Topological Query**: Use `get_elements_by_room` to retrieve the IFC graph subset.
    3. **Semantic Filtering**: Reason to find the specific element.
    4. **Verification**: Use `get_element_details` if needed.
    5. **Output**: Return a structured summary including the unique GUID.

    Constraints:
    - If you cannot find the room, ask for clarification.
    - Always output the GUID if identified.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 3. 组装 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 4. 加载并执行测试剧本
    scenarios = load_scenarios()
    
    print(f"{Fore.CYAN}=== Agent initialized. Loaded {len(scenarios)} scenarios. ==={Style.RESET_ALL}")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{Fore.WHITE}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Running Scenario {i}: {scenario['name']}{Style.RESET_ALL}")
        print(f"{Style.DIM}Description: {scenario['description']}{Style.RESET_ALL}")
        
        user_input = scenario['input']
        print(f"\n{Fore.YELLOW}>>> Input: {user_input}{Style.RESET_ALL}")
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            response = agent_executor.invoke({"input": user_input})
            
            elapsed = time.time() - start_time
            
            print(f"\n{Fore.GREEN}>>> Final Response ({elapsed:.2f}s):{Style.RESET_ALL}")
            print(response['output'])
            
        except Exception as e:
            print(f"{Fore.RED}Error during execution: {e}")
        
        # 演示时的戏剧性停顿（可选，方便面试官看清楚输出）
        if i < len(scenarios):
            print(f"\n{Fore.BLUE}...Proceeding to next scenario in 3 seconds...{Style.RESET_ALL}")
            time.sleep(3)

if __name__ == "__main__":
    main()