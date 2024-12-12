#conversation_analytics.py
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict

from plugins.plugin import Plugin

class ConversationAnalyticsPlugin(Plugin):
    """
    Plugin for analyzing conversation patterns and generating insights from chat history
    """
    def __init__(self):
        self.analytics_dir = os.path.join(os.path.dirname(__file__), 'analytics')
        os.makedirs(self.analytics_dir, exist_ok=True)
        self.analytics_file = os.path.join(self.analytics_dir, 'conversation_stats.json')
        self.conversation_stats = self.load_stats()

    def get_source_name(self) -> str:
        return "ConversationAnalytics"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "analyze_conversation",
            "description": "Analyze conversation patterns and generate insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "ID of chat to analyze"
                    },
                    "time_period": {
                        "type": "string",
                        "description": "Time period to analyze",
                        "enum": ["day", "week", "month"]
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis to perform",
                        "enum": ["usage", "topics", "sentiment", "all"]
                    }
                },
                "required": ["chat_id", "time_period", "analysis_type"]
            }
        },
        {
            "name": "get_personalized_recommendations",
            "description": "Generate personalized recommendations based on user interaction history",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "ID of chat to analyze"
                    },
                    "recommendation_type": {
                        "type": "string",
                        "description": "Type of recommendations to generate",
                        "enum": ["topics", "learning", "content_format", "interaction_style"]
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of recommendations to generate",
                        "default": 3
                    }
                },
                "required": ["chat_id", "recommendation_type"]
            }
        }]

    def load_stats(self) -> Dict:
        """Load analytics data from file"""
        def create_default_stats():
            # Initialize stats with all hours set to 0 and token_usage as defaultdict
            return {
                'messages': [],
                'token_usage': defaultdict(int),  # Changed from regular dict to defaultdict
                'topics': defaultdict(int),
                'sentiment_scores': [],
                'active_hours': {str(hour): 0 for hour in range(24)}  # Initialize all 24 hours
            }

        if os.path.exists(self.analytics_file):
            try:
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    # Convert to defaultdict and ensure all required structures exist
                    for chat_id in stats:
                        # Convert token_usage to defaultdict
                        stats[chat_id]['token_usage'] = defaultdict(
                            int, stats[chat_id].get('token_usage', {})
                        )
                        # Ensure topics is a defaultdict
                        stats[chat_id]['topics'] = defaultdict(
                            int, stats[chat_id].get('topics', {})
                        )
                        # Ensure active_hours exists with all hours
                        if 'active_hours' not in stats[chat_id]:
                            stats[chat_id]['active_hours'] = {str(hour): 0 for hour in range(24)}
                        else:
                            # Ensure all hours exist in existing stats
                            for hour in range(24):
                                if str(hour) not in stats[chat_id]['active_hours']:
                                    stats[chat_id]['active_hours'][str(hour)] = 0
                        # Ensure other required fields exist
                        if 'messages' not in stats[chat_id]:
                            stats[chat_id]['messages'] = []
                        if 'sentiment_scores' not in stats[chat_id]:
                            stats[chat_id]['sentiment_scores'] = []
                    return stats
            except Exception as e:
                logging.error(f"Failed to load conversation stats: {e}")
                return defaultdict(create_default_stats)
        return defaultdict(create_default_stats)

    def save_stats(self):
        """Save analytics data to file"""
        try:
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save conversation stats: {e}")

    def update_stats(self, chat_id: str, message_data: Dict):
        """Update conversation statistics with new message data"""
        # Initialize stats for new chat_id if it doesn't exist
        if chat_id not in self.conversation_stats:
            self.conversation_stats[chat_id] = {
                'messages': [],
                'token_usage': defaultdict(int),
                'topics': defaultdict(int),
                'sentiment_scores': [],
                'active_hours': {str(hour): 0 for hour in range(24)}
            }
        
        stats = self.conversation_stats[chat_id]
        
        # Add message to history
        message_data['timestamp'] = datetime.now().isoformat()
        stats['messages'].append(message_data)
        
        # Update token usage
        if 'tokens' in message_data:
            stats['token_usage'][datetime.now().strftime('%Y-%m-%d')] += message_data['tokens']
        
        # Update active hours
        hour = datetime.now().hour
        stats['active_hours'][str(hour)] += 1

        # Cleanup old data (keep last 30 days)
        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
        stats['messages'] = [
            msg for msg in stats['messages'] 
            if msg['timestamp'] > cutoff_date
        ]

        self.save_stats()

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """Execute plugin functions"""
        if function_name == "get_personalized_recommendations":
            chat_id = kwargs['chat_id']
            recommendation_type = kwargs['recommendation_type']
            count = kwargs.get('count', 3)

            if chat_id not in self.conversation_stats:
                return {
                    "error": "No conversation data found for this chat"
                }

            stats = self.conversation_stats[chat_id]
            
            # Get recent messages for analysis
            recent_messages = stats['messages'][-100:]  # Анализируем последние 100 сообщений
            
            recommendations = []
            
            if recommendation_type == "topics":
                # Анализ частых тем и предложение связанных тем
                topics_counter = defaultdict(int)
                for msg in recent_messages:
                    if 'topics' in msg:
                        for topic in msg['topics']:
                            topics_counter[topic] += 1
                
                prompt = f"""Based on the following frequently discussed topics:
                {dict(sorted(topics_counter.items(), key=lambda x: x[1], reverse=True)[:5])}
                
                Suggest {count} new related topics that might interest the user. For each topic:
                1. Provide a brief explanation why it's relevant
                2. Suggest a starting question for this topic
                3. List potential learning opportunities"""
                
                response, _ = await helper.get_chat_response(
                    chat_id=hash(f"{chat_id}_topics"),
                    query=prompt
                )
                recommendations.append(("Related Topics", response))

            elif recommendation_type == "learning":
                # Анализ сложности взаимодействий и предложение путей обучения
                avg_msg_length = sum(len(msg.get('text', '')) for msg in recent_messages) / len(recent_messages)
                technical_terms = self._extract_technical_terms(recent_messages)
                
                prompt = f"""Based on user's conversation history:
                - Average message length: {avg_msg_length:.0f} characters
                - Technical terms used: {', '.join(technical_terms[:5])}
                
                Suggest {count} personalized learning paths. For each:
                1. Identify current knowledge level
                2. Suggest next learning steps
                3. Recommend specific resources or exercises"""
                
                response, _ = await helper.get_chat_response(
                    chat_id=hash(f"{chat_id}_learning"),
                    query=prompt
                )
                recommendations.append(("Learning Paths", response))

            elif recommendation_type == "content_format":
                # Анализ предпочтительных форматов контента
                format_preferences = self._analyze_format_preferences(recent_messages)
                
                prompt = f"""Based on user's format preferences:
                {format_preferences}
                
                Suggest {count} ways to optimize content delivery. Include:
                1. Preferred content formats
                2. Optimal content length
                3. Presentation style suggestions"""
                
                response, _ = await helper.get_chat_response(
                    chat_id=hash(f"{chat_id}_format"),
                    query=prompt
                )
                recommendations.append(("Content Format", response))

            elif recommendation_type == "interaction_style":
                # Анализ стиля взаимодействия
                interaction_patterns = self._analyze_interaction_patterns(recent_messages)
                
                prompt = f"""Based on user's interaction patterns:
                {interaction_patterns}
                
                Suggest {count} ways to improve interaction. Include:
                1. Communication style preferences
                2. Response format recommendations
                3. Engagement optimization tips"""
                
                response, _ = await helper.get_chat_response(
                    chat_id=hash(f"{chat_id}_style"),
                    query=prompt
                )
                recommendations.append(("Interaction Style", response))

            return {
                "recommendations": recommendations,
                "based_on_messages": len(recent_messages),
                "recommendation_type": recommendation_type
            }
        elif function_name == "analyze_conversation":
            chat_id = kwargs['chat_id']
            time_period = kwargs['time_period']
            analysis_type = kwargs.get('analysis_type', 'all')

            if chat_id not in self.conversation_stats:
                return {
                    "error": "No conversation data found for this chat"
                }

            stats = self.conversation_stats[chat_id]
            
            # Calculate time period cutoff
            now = datetime.now()
            if time_period == 'day':
                cutoff = now - timedelta(days=1)
            elif time_period == 'week':
                cutoff = now - timedelta(weeks=1)
            else:  # month
                cutoff = now - timedelta(days=30)

            cutoff = cutoff.isoformat()

            # Filter messages within time period
            period_messages = [
                msg for msg in stats['messages']
                if msg['timestamp'] > cutoff
            ]

            results = {}

            if analysis_type in ['usage', 'all']:
                # Calculate usage statistics
                total_messages = len(period_messages)
                total_tokens = sum(
                    msg.get('tokens', 0) for msg in period_messages
                )
                avg_tokens_per_message = total_tokens / total_messages if total_messages > 0 else 0

                results['usage_stats'] = {
                    'total_messages': total_messages,
                    'total_tokens': total_tokens,
                    'avg_tokens_per_message': round(avg_tokens_per_message, 2)
                }

            if analysis_type in ['topics', 'all']:
                # Get recent topics/themes
                topics = defaultdict(int)
                for msg in period_messages:
                    if 'topics' in msg:
                        for topic in msg['topics']:
                            topics[topic] += 1

                results['topic_analysis'] = {
                    'most_common_topics': dict(
                        sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
                    )
                }

            if analysis_type in ['sentiment', 'all']:
                # Calculate activity patterns
                hour_activity = defaultdict(int)
                for msg in period_messages:
                    hour = datetime.fromisoformat(msg['timestamp']).hour
                    hour_activity[str(hour)] += 1

                results['activity_patterns'] = {
                    'hourly_activity': dict(sorted(hour_activity.items())),
                    'peak_hours': sorted(
                        hour_activity.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                }

            # Format response
            analysis_text = [
                f"📊 Conversation Analysis ({time_period})\n"
            ]

            if 'usage_stats' in results:
                analysis_text.extend([
                    "\n📝 Usage Statistics:",
                    f"• Total Messages: {results['usage_stats']['total_messages']}",
                    f"• Total Tokens: {results['usage_stats']['total_tokens']}",
                    f"• Avg Tokens/Message: {results['usage_stats']['avg_tokens_per_message']}"
                ])

            if 'topic_analysis' in results:
                analysis_text.extend([
                    "\n🎯 Common Topics:"
                ])
                for topic, count in results['topic_analysis']['most_common_topics'].items():
                    analysis_text.append(f"• {topic}: {count} mentions")

            if 'activity_patterns' in results:
                analysis_text.extend([
                    "\n⏰ Peak Activity Hours:"
                ])
                for hour, count in results['activity_patterns']['peak_hours']:
                    analysis_text.append(f"• {hour}:00 - {count} messages")

            return {
                "result": "\n".join(analysis_text)
            }

        return {"error": "Unknown function"}
    def _extract_technical_terms(self, messages: List[Dict]) -> List[str]:
            """Извлекает технические термины из сообщений"""
            technical_terms = set()
            
            # Предопределенный список технических терминов
            TECHNICAL_TERMS = {
                'api', 'token', 'database', 'server', 'client',
                'http', 'https', 'rest', 'json', 'xml',
                'sql', 'nosql', 'cache', 'async', 'sync',
                'frontend', 'backend', 'api key', 'endpoint',
                'request', 'response', 'webhook', 'cors',
                # Добавьте другие технические термины по необходимости
            }
            
            for msg in messages:
                if not msg.get('text'):
                    continue
                    
                text = msg['text'].lower()
                words = set(text.split())
                
                # Поиск одиночных слов
                technical_terms.update(words & TECHNICAL_TERMS)
                
                # Поиск составных терминов
                for term in TECHNICAL_TERMS:
                    if ' ' in term and term in text:
                        technical_terms.add(term)
            
            return sorted(list(technical_terms))
    
    def _analyze_format_preferences(self, messages: List[Dict]) -> Dict:
        """Анализирует предпочтения по формату контента"""
        formats = {
            'text': 0,
            'code': 0,
            'images': 0,
            'voice': 0
        }
        
        for msg in messages:
            if 'text' in msg: formats['text'] += 1
            if 'code' in str(msg.get('text', '')): formats['code'] += 1
            if msg.get('has_image'): formats['images'] += 1
            if msg.get('has_voice'): formats['voice'] += 1
            
        return formats

    def _analyze_interaction_patterns(self, messages: List[Dict]) -> Dict:
        """Анализирует паттерны взаимодействия"""
        patterns = {
            'avg_response_time': 0,
            'message_length': [],
            'question_frequency': 0,
            'command_usage': 0
        }
        
        for i, msg in enumerate(messages):
            patterns['message_length'].append(len(msg.get('text', '')))
            if '?' in msg.get('text', ''): 
                patterns['question_frequency'] += 1
            if msg.get('is_command'): 
                patterns['command_usage'] += 1
            
        if patterns['message_length']:
            patterns['avg_message_length'] = sum(patterns['message_length']) / len(patterns['message_length'])
            
        return patterns