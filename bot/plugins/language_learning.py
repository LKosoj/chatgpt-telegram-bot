#language_learning.py
import os
import json
import random
from datetime import datetime
from typing import Dict, List

from .plugin import Plugin

class LanguageLearningPlugin(Plugin):
    """
    A plugin for language learning features with daily exercises and progress tracking
    """
    def __init__(self):
        self.exercises_dir = os.path.join(os.path.dirname(__file__), 'language_data')
        self.users_progress_file = os.path.join(os.path.dirname(__file__), 'language_progress.json')
        self.supported_languages = {
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'russian': 'ru'
        }
        # Create directories if they don't exist
        os.makedirs(self.exercises_dir, exist_ok=True)
        self.load_progress()

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        if storage_root:
            self.exercises_dir = os.path.join(storage_root, 'language_data')
            self.users_progress_file = os.path.join(storage_root, 'language_progress.json')
            os.makedirs(self.exercises_dir, exist_ok=True)
            self.load_progress()

    def get_source_name(self) -> str:
        return "LanguageLearning"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "daily_practice",
            "description": "Generate daily language practice exercises",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Target language to practice",
                        "enum": list(self.supported_languages.keys())
                    },
                    "level": {
                        "type": "string",
                        "description": "Proficiency level",
                        "enum": ["beginner", "intermediate", "advanced"]
                    },
                    "exercise_type": {
                        "type": "string",
                        "description": "Type of exercise to practice",
                        "enum": ["vocabulary", "grammar", "conversation"]
                    }
                },
                "required": ["language", "level"]
            }
        }, {
            "name": "track_progress",
            "description": "Track user's language learning progress",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Language to track progress for",
                        "enum": list(self.supported_languages.keys())
                    },
                    "completed_exercise": {
                        "type": "boolean",
                        "description": "Whether the exercise was completed successfully"
                    }
                },
                "required": ["language", "completed_exercise"]
            }
        }]

    def load_progress(self):
        """Load user progress from file"""
        if os.path.exists(self.users_progress_file):
            with open(self.users_progress_file, 'r', encoding='utf-8') as f:
                self.users_progress = json.load(f)
        else:
            self.users_progress = {}

    def save_progress(self):
        """Save user progress to file"""
        with open(self.users_progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.users_progress, f, ensure_ascii=False, indent=2)

    def get_exercise_prompt(self, language: str, level: str, exercise_type: str) -> str:
        """Generate exercise prompt based on parameters"""
        prompts = {
            "vocabulary": {
                "beginner": [
                    "Basic everyday objects and actions",
                    "Numbers and colors",
                    "Family members and professions"
                ],
                "intermediate": [
                    "Weather and environment",
                    "Travel and transportation",
                    "Food and dining"
                ],
                "advanced": [
                    "Business and economics",
                    "Science and technology",
                    "Arts and culture"
                ]
            },
            "grammar": {
                "beginner": [
                    "Present tense conjugation",
                    "Basic pronouns and articles",
                    "Simple questions and answers"
                ],
                "intermediate": [
                    "Past tense usage",
                    "Conditional statements",
                    "Comparative and superlative"
                ],
                "advanced": [
                    "Complex verb tenses",
                    "Subjunctive mood",
                    "Idiomatic expressions"
                ]
            },
            "conversation": {
                "beginner": [
                    "Introducing yourself",
                    "Ordering food",
                    "Asking for directions"
                ],
                "intermediate": [
                    "Making appointments",
                    "Discussing hobbies",
                    "Shopping and bargaining"
                ],
                "advanced": [
                    "Debating current events",
                    "Professional interviews",
                    "Cultural discussions"
                ]
            }
        }

        selected_topic = random.choice(prompts[exercise_type][level])
        return selected_topic

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """Execute plugin functions"""
        if function_name == "daily_practice":
            language = kwargs.get('language').lower()
            level = kwargs.get('level')
            exercise_type = kwargs.get('exercise_type', 'vocabulary')
            
            if language not in self.supported_languages:
                return {"error": f"Language {language} is not supported"}

            # Get the exercise topic
            topic = self.get_exercise_prompt(language, level, exercise_type)
            
            # Generate exercise using ChatGPT
            prompt = (
                f"Create a {level} level {exercise_type} exercise in {language} about {topic}. "
                f"Include:\n1. Instructions in English\n2. Exercise content in {language}\n"
                f"3. Correct answers or example responses\n4. Additional tips for learning"
            )
            
            # Use the helper to get response from ChatGPT
            response, _ = await helper.get_chat_response(
                chat_id=hash(f"{language}_{level}_{exercise_type}"),
                query=prompt
            )

            return {
                "exercise": {
                    "language": language,
                    "level": level,
                    "type": exercise_type,
                    "topic": topic,
                    "content": response
                }
            }

        elif function_name == "track_progress":
            user_id = str(helper.user_id)
            language = kwargs.get('language').lower()
            completed = kwargs.get('completed_exercise')
            
            if user_id not in self.users_progress:
                self.users_progress[user_id] = {}
            
            if language not in self.users_progress[user_id]:
                self.users_progress[user_id][language] = {
                    "exercises_completed": 0,
                    "last_practice": None,
                    "streak": 0
                }

            today = datetime.now().strftime("%Y-%m-%d")
            user_lang_progress = self.users_progress[user_id][language]

            if completed:
                user_lang_progress["exercises_completed"] += 1
                
                # Update streak
                if user_lang_progress["last_practice"] != today:
                    last_date = datetime.strptime(user_lang_progress["last_practice"], "%Y-%m-%d") \
                        if user_lang_progress["last_practice"] else None
                    
                    if last_date and (datetime.now() - last_date).days == 1:
                        user_lang_progress["streak"] += 1
                    else:
                        user_lang_progress["streak"] = 1
                        
                user_lang_progress["last_practice"] = today
                
            self.save_progress()
            
            return {
                "progress": {
                    "language": language,
                    "total_completed": user_lang_progress["exercises_completed"],
                    "current_streak": user_lang_progress["streak"],
                    "last_practice": user_lang_progress["last_practice"]
                }
            }

        return {"error": "Unknown function"}
