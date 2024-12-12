#task_management.py
import os
import json
from datetime import datetime
from typing import Dict, Any
from plugins.plugin import Plugin

class TaskManagementPlugin(Plugin):
    """
    Plugin for managing tasks and tracking their status
    """
    def __init__(self):
        self.tasks = {}
        self.tasks_file = os.path.join(os.path.dirname(__file__), "tasks.json")
        self.load_tasks()

    def get_source_name(self) -> str:
        return "TaskManagement"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "create_task",
            "description": "Create a new task with priority and deadline",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Task title"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed task description"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Task priority level",
                        "enum": ["high", "medium", "low"]
                    },
                    "deadline": {
                        "type": "string",
                        "description": "Task deadline in format YYYY-MM-DD HH:MM"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags for the task"
                    }
                },
                "required": ["title", "priority"]
            }
        }, {
            "name": "list_tasks",
            "description": "List all tasks with optional filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by task status",
                        "enum": ["pending", "in_progress", "completed"]
                    },
                    "priority": {
                        "type": "string",
                        "description": "Filter by priority level",
                        "enum": ["high", "medium", "low"]
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by specific tag"
                    }
                }
            }
        }, {
            "name": "update_task",
            "description": "Update task status or details",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the task to update"
                    },
                    "status": {
                        "type": "string",
                        "description": "New task status",
                        "enum": ["pending", "in_progress", "completed"]
                    },
                    "priority": {
                        "type": "string",
                        "description": "New priority level",
                        "enum": ["high", "medium", "low"]
                    },
                    "description": {
                        "type": "string",
                        "description": "Updated task description"
                    }
                },
                "required": ["task_id"]
            }
        }]

    def load_tasks(self):
        """Load tasks from file"""
        if os.path.exists(self.tasks_file):
            with open(self.tasks_file, 'r', encoding='utf-8') as f:
                self.tasks = json.load(f)

    def save_tasks(self):
        """Save tasks to file"""
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, ensure_ascii=False, indent=2)

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """Execute plugin functions"""
        user_id = str(helper.user_id)
        
        if function_name == "create_task":
            task_id = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{user_id}'
            
            task = {
                "id": task_id,
                "user_id": user_id,
                "title": kwargs["title"],
                "description": kwargs.get("description", ""),
                "priority": kwargs["priority"],
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "deadline": kwargs.get("deadline"),
                "tags": kwargs.get("tags", []),
                "last_updated": datetime.now().isoformat()
            }

            if user_id not in self.tasks:
                self.tasks[user_id] = {}
            self.tasks[user_id][task_id] = task
            self.save_tasks()
            
            return {
                "success": True,
                "message": f"Task created successfully with ID: {task_id}",
                "task": task
            }

        elif function_name == "list_tasks":
            user_tasks = self.tasks.get(user_id, {})
            
            if not user_tasks:
                return {"message": "No tasks found"}
            
            # Apply filters
            filtered_tasks = user_tasks.values()
            if "status" in kwargs:
                filtered_tasks = [t for t in filtered_tasks if t["status"] == kwargs["status"]]
            if "priority" in kwargs:
                filtered_tasks = [t for t in filtered_tasks if t["priority"] == kwargs["priority"]]
            if "tag" in kwargs:
                filtered_tasks = [t for t in filtered_tasks if kwargs["tag"] in t["tags"]]

            # Format tasks for display
            tasks_list = []
            for task in filtered_tasks:
                task_info = [
                    f"📋 Task ID: {task['id']}",
                    f"📌 Title: {task['title']}",
                    f"📝 Description: {task['description']}" if task['description'] else "",
                    f"🎯 Priority: {task['priority'].upper()}",
                    f"📊 Status: {task['status'].replace('_', ' ').title()}",
                    f"⏰ Deadline: {task['deadline']}" if task['deadline'] else "",
                    f"🏷️ Tags: {', '.join(task['tags'])}" if task['tags'] else ""
                ]
                tasks_list.append("\n".join(filter(None, task_info)))

            return {
                "message": "Your tasks:\n\n" + "\n\n".join(tasks_list)
            }

        elif function_name == "update_task":
            task_id = kwargs["task_id"]
            if user_id not in self.tasks or task_id not in self.tasks[user_id]:
                return {"error": "Task not found"}

            task = self.tasks[user_id][task_id]
            
            # Update task fields
            if "status" in kwargs:
                task["status"] = kwargs["status"]
            if "priority" in kwargs:
                task["priority"] = kwargs["priority"]
            if "description" in kwargs:
                task["description"] = kwargs["description"]
                
            task["last_updated"] = datetime.now().isoformat()
            self.save_tasks()

            return {
                "success": True,
                "message": f"Task {task_id} updated successfully",
                "task": task
            }

        return {"error": "Unknown function"}