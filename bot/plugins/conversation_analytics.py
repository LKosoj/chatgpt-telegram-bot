#conversation_analytics.py
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict

from .plugin import Plugin

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
        }]

    def load_stats(self) -> Dict:
        """Load analytics data from file"""
        if os.path.exists(self.analytics_file):
            with open(self.analytics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return defaultdict(lambda: {
            'messages': [],
            'token_usage': defaultdict(int),
            'topics': defaultdict(int),
            'sentiment_scores': [],
            'active_hours': defaultdict(int)
        })

    def save_stats(self):
        """Save analytics data to file"""
        with open(self.analytics_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_stats, f, ensure_ascii=False, indent=2)

    def update_stats(self, chat_id: str, message_data: Dict):
        """Update conversation statistics with new message data"""
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
        if function_name == "analyze_conversation":
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