import json
import os
from datetime import datetime

class ConversationLogger:
    """Logs conversations between user and agent to JSON file"""

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"conversation_{timestamp}.json")

        self.conversation = {
            "session_start": datetime.now().isoformat(),
            "messages": []
        }

        print(f"📝 Logging conversation to: {self.log_file}")

    def log_user_message(self, message: str):
        """Log a user message"""
        self.conversation["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "role": "user",
            "content": message
        })
        self._save()

    def log_agent_message(self, message: str, tool_calls=None):
        """Log an agent response with optional tool calls"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": "assistant",
            "content": message
        }

        if tool_calls:
            entry["tool_calls"] = tool_calls

        self.conversation["messages"].append(entry)
        self._save()

    def log_tool_call(self, tool_name: str, args: dict, result: str):
        """Log a tool call and its result"""
        self.conversation["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "role": "tool",
            "tool_name": tool_name,
            "arguments": args,
            "result": result
        })
        self._save()

    def _save(self):
        """Save conversation to file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversation, f, indent=2, ensure_ascii=False)

    def save_summary(self, summary_text: str):
        """Save a summary of the conversation"""
        self.conversation["summary"] = summary_text
        self.conversation["session_end"] = datetime.now().isoformat()
        self._save()
        print(f"✅ Conversation saved to: {self.log_file}")