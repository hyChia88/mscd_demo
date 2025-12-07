import os
import yaml
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

from agent_tools import tools

def load_system_prompt(prompt_file="prompts/system_prompt.yaml"):
    """Load system prompt from YAML file"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, prompt_file)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("system_prompt", "")

def main():
    """
    Interactive CLI Chat Interface for BIM Inspection Agent

    Usage:
        python src/chat_cli.py

    Commands:
        - Type your question and press Enter
        - Type 'quit', 'exit', or 'q' to exit
        - Type 'clear' to clear conversation history
    """

    print("=" * 70)
    print("🏗️  BIM Inspection Agent - Interactive Chat")
    print("=" * 70)
    print("\nInitializing agent...")

    # 1. Setup LLM
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

    # 3. Create Agent
    agent_executor = create_react_agent(
        llm.bind(system=system_prompt),
        tools
    )

    print("✅ Agent ready!\n")
    print("Type your questions below. Commands: 'quit' to exit, 'clear' to reset.\n")
    print("-" * 70)

    # Conversation history
    conversation_history = []

    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == 'clear':
                conversation_history = []
                print("\n🔄 Conversation history cleared.")
                continue

            if not user_input:
                continue

            # Add user message to history
            conversation_history.append(("user", user_input))

            # Get agent response
            print("\n🤖 Agent: ", end="", flush=True)

            response = agent_executor.invoke({"messages": conversation_history})

            # Extract and display response
            if "messages" in response:
                agent_message = response["messages"][-1].content
                conversation_history = response["messages"]
            else:
                agent_message = str(response)

            print(agent_message)
            print("\n" + "-" * 70)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()
