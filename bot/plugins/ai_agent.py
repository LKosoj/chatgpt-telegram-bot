from typing import Optional, List, Dict, Any
import json
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from queue import Queue, PriorityQueue
from dataclasses import dataclass
from enum import Enum, IntEnum
from datetime import datetime, timedelta
import sqlite3
import aiosqlite
import uuid
import os
import pathlib
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import List, Dict
import spacy
from transformers import pipeline
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from textblob import TextBlob
import asyncpg
import uuid
import os
import heapq

class AgentRole(Enum):
    RESEARCHER = "researcher"
    PLANNER = "planner"
    EXECUTOR = "executor"

class TaskPriority(IntEnum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

@dataclass
class AgentMessage:
    from_role: AgentRole
    to_role: AgentRole
    content: str
    metadata: Dict = None

class PlanStep:
    def __init__(self, action: str, description: str, estimated_time: int = None):
        self.action = action
        self.description = description
        self.estimated_time = estimated_time
        self.status = "pending"
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "description": self.description,
            "estimated_time": self.estimated_time,
            "status": self.status,
            "created_at": self.created_at
        }

class ActionPlan:
    def __init__(self, goal: str, context: str = None):
        self.goal = goal
        self.context = context
        self.steps: List[PlanStep] = []
        self.created_at = datetime.now().isoformat()
        self.status = "created"
        
    def add_step(self, action: str, description: str, estimated_time: int = None):
        step = PlanStep(action, description, estimated_time)
        self.steps.append(step)
        
    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "context": self.context,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
            "status": self.status,
            "total_steps": len(self.steps)
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

class ResearchResult:
    def __init__(self, query: str):
        self.query = query
        self.sources: List[Dict[str, str]] = []
        self.analysis: Dict[str, Any] = {}
        self.summary: str = ""
        self.created_at = datetime.now().isoformat()
        
    def add_source(self, title: str, content: str, url: str = None):
        self.sources.append({
            "title": title,
            "content": content,
            "url": url,
            "added_at": datetime.now().isoformat()
        })
        
    def set_analysis(self, analysis_data: Dict[str, Any]):
        self.analysis = analysis_data
        
    def set_summary(self, summary: str):
        self.summary = summary
        
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "sources": self.sources,
            "analysis": self.analysis,
            "summary": self.summary,
            "created_at": self.created_at
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

class NLPAnalyzer:
    def __init__(self):
        # Загружаем модели
        self.nlp = spacy.load("ru_core_news_lg")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Комплексный анализ текста"""
        doc = self.nlp(text)
        
        # Базовый анализ
        basic_analysis = {
            "sentences_count": len(list(doc.sents)),
            "words_count": len([token for token in doc if not token.is_punct]),
            "characters_count": len(text)
        }
        
        # Морфологический анализ
        morphological_analysis = self._analyze_morphology(doc)
        
        # Синтаксический анализ
        syntactic_analysis = self._analyze_syntax(doc)
        
        # Семантический анализ
        semantic_analysis = self._analyze_semantics(doc)
        
        # Анализ тональности
        sentiment_analysis = self._analyze_sentiment(text)
        
        # Извлечение ключевых фраз
        key_phrases = self._extract_key_phrases(doc)
        
        # Тематический анализ
        topic_analysis = self._analyze_topics(text)
        
        return {
            "basic_analysis": basic_analysis,
            "morphological_analysis": morphological_analysis,
            "syntactic_analysis": syntactic_analysis,
            "semantic_analysis": semantic_analysis,
            "sentiment_analysis": sentiment_analysis,
            "key_phrases": key_phrases,
            "topic_analysis": topic_analysis
        }
    
    def _analyze_morphology(self, doc) -> Dict[str, Any]:
        """Морфологический анализ текста"""
        pos_counts = Counter([token.pos_ for token in doc])
        
        return {
            "pos_distribution": dict(pos_counts),
            "lemmas": [token.lemma_ for token in doc if not token.is_punct],
            "unique_lemmas_count": len(set(token.lemma_ for token in doc if not token.is_punct))
        }
    
    def _analyze_syntax(self, doc) -> Dict[str, Any]:
        """Синтаксический анализ текста"""
        return {
            "dependencies": [
                {
                    "word": token.text,
                    "dependency": token.dep_,
                    "head": token.head.text
                }
                for token in doc
            ],
            "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
            "sentence_structures": [
                {
                    "text": sent.text,
                    "root": sent.root.text,
                    "subject": next((token.text for token in sent if token.dep_ == "nsubj"), None)
                }
                for sent in doc.sents
            ]
        }
    
    def _analyze_semantics(self, doc) -> Dict[str, Any]:
        """Семантический анализ текста"""
        # Извлекаем именованные сущности
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_)
            }
            for ent in doc.ents
        ]
        
        # Анализируем семантическую близость слов
        semantic_similarities = []
        words = [token for token in doc if token.has_vector and not token.is_punct]
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                similarity = word1.similarity(word2)
                if similarity > 0.5:  # Отбираем только значимые связи
                    semantic_similarities.append({
                        "word1": word1.text,
                        "word2": word2.text,
                        "similarity": float(similarity)
                    })
        
        return {
            "entities": entities,
            "semantic_similarities": semantic_similarities
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализ тональности текста"""
        # Используем несколько подходов для более точного анализа
        
        # 1. Используем RuBERT для анализа тональности
        rubert_sentiment = self.sentiment_analyzer(text)[0]
        
        # 2. Используем TextBlob как дополнительный источник
        blob = TextBlob(text)
        textblob_sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
        
        # 3. Анализируем тональность по предложениям
        sentences = [str(sent) for sent in self.nlp(text).sents]
        sentence_sentiments = [
            {
                "text": sent,
                "sentiment": self.sentiment_analyzer(sent)[0]
            }
            for sent in sentences
        ]
        
        return {
            "overall_sentiment": rubert_sentiment,
            "textblob_sentiment": textblob_sentiment,
            "sentence_sentiments": sentence_sentiments
        }
    
    def _extract_key_phrases(self, doc) -> List[Dict[str, Any]]:
        """Извлечение ключевых фраз"""
        # Используем комбинацию подходов для извлечения ключевых фраз
        
        # 1. Извлекаем именные группы
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # 2. Используем статистические метрики
        words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        word_freq = Counter(words)
        
        # 3. Используем TF-IDF для оценки важности фраз
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10)
        try:
            tfidf_matrix = vectorizer.fit_transform([doc.text])
            feature_names = vectorizer.get_feature_names_out()
            scores = zip(feature_names, tfidf_matrix.toarray()[0])
            tfidf_phrases = sorted(scores, key=lambda x: x[1], reverse=True)
        except:
            tfidf_phrases = []
        
        return {
            "noun_phrases": noun_phrases[:10],
            "frequent_words": dict(word_freq.most_common(10)),
            "tfidf_phrases": [
                {"phrase": phrase, "score": float(score)}
                for phrase, score in tfidf_phrases[:10]
            ]
        }
    
    def _analyze_topics(self, text: str) -> Dict[str, Any]:
        """Тематический анализ текста"""
        # Определяем возможные темы
        candidate_topics = [
            "технологии", "наука", "бизнес", "политика", 
            "культура", "спорт", "образование", "медицина"
        ]
        
        # Используем zero-shot классификацию для определения тем
        topic_results = self.zero_shot_classifier(
            text,
            candidate_topics,
            multi_label=True
        )
        
        # Создаем краткое содержание текста
        try:
            summary = self.summarizer(
                text,
                max_length=130,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]
        except:
            summary = "Не удалось создать краткое содержание"
        
        return {
            "detected_topics": [
                {
                    "topic": label,
                    "confidence": float(score)
                }
                for label, score in zip(topic_results["labels"], topic_results["scores"])
                if score > 0.3  # Отбираем только темы с высокой уверенностью
            ],
            "summary": summary
        }

class ResearchStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    async def initialize(self):
        """Инициализация базы данных и создание таблиц"""
        async with aiosqlite.connect(self.db_path) as db:
            # Создаем таблицу для исследований
            await db.execute('''
                CREATE TABLE IF NOT EXISTS researches (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT,
                    user_id TEXT
                )
            ''')
            
            # Создаем таблицу для источников
            await db.execute('''
                CREATE TABLE IF NOT EXISTS research_sources (
                    id TEXT PRIMARY KEY,
                    research_id TEXT,
                    title TEXT,
                    content TEXT,
                    url TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (research_id) REFERENCES researches(id)
                )
            ''')
            
            # Создаем таблицу для результатов анализа
            await db.execute('''
                CREATE TABLE IF NOT EXISTS research_analysis (
                    id TEXT PRIMARY KEY,
                    research_id TEXT,
                    analysis_type TEXT NOT NULL,
                    analysis_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (research_id) REFERENCES researches(id)
                )
            ''')
            
            # Создаем таблицу для связей между исследованиями
            await db.execute('''
                CREATE TABLE IF NOT EXISTS research_relations (
                    id TEXT PRIMARY KEY,
                    source_research_id TEXT,
                    target_research_id TEXT,
                    relation_type TEXT NOT NULL,
                    similarity_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (source_research_id) REFERENCES researches(id),
                    FOREIGN KEY (target_research_id) REFERENCES researches(id)
                )
            ''')
            
            # Создаем индексы для оптимизации поиска связей
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_research_relations_source 
                ON research_relations(source_research_id)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_research_relations_target 
                ON research_relations(target_research_id)
            ''')
            
            await db.commit()

    async def save_research(self, research: ResearchResult, user_id: str = None) -> str:
        """Сохранение результатов исследования в базу данных"""
        research_id = str(uuid.uuid4())
        
        async with aiosqlite.connect(self.db_path) as db:
            # Сохраняем основную информацию об исследовании
            await db.execute('''
                INSERT INTO researches (id, query, summary, user_id)
                VALUES (?, ?, ?, ?)
            ''', (research_id, research.query, research.summary, user_id))
            
            # Сохраняем источники
            for source in research.sources:
                source_id = str(uuid.uuid4())
                await db.execute('''
                    INSERT INTO research_sources (id, research_id, title, content, url)
                    VALUES (?, ?, ?, ?, ?)
                ''', (source_id, research_id, source['title'], 
                     source['content'], source.get('url')))
            
            # Сохраняем результаты анализа
            if research.analysis:
                analysis_id = str(uuid.uuid4())
                await db.execute('''
                    INSERT INTO research_analysis (id, research_id, analysis_type, analysis_data)
                    VALUES (?, ?, ?, ?)
                ''', (analysis_id, research_id, 'general', 
                     json.dumps(research.analysis, ensure_ascii=False)))
            
            # Поиск и создание связей с похожими исследованиями
            cursor = await db.execute('''
                SELECT id, query, summary FROM researches 
                WHERE id != ? ORDER BY created_at DESC LIMIT 50
            ''', (research_id,))
            
            existing_researches = await cursor.fetchall()
            for existing in existing_researches:
                # Вычисляем схожесть между исследованиями
                similarity = await self._calculate_similarity(
                    research.query, research.summary,
                    existing['query'], existing['summary']
                )
                
                if similarity > 0.3:  # Создаем связь только если схожесть выше порога
                    relation_type = "similar" if similarity > 0.7 else "related"
                    await self.create_research_relation(
                        research_id,
                        existing['id'],
                        relation_type,
                        similarity,
                        {
                            "common_topics": await self._find_common_topics(
                                research.summary, existing['summary']
                            )
                        }
                    )
        
            await db.commit()
        
        return research_id

    async def get_research(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Получение результатов исследования из базы данных"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Получаем основную информацию
            cursor = await db.execute('''
                SELECT * FROM researches WHERE id = ?
            ''', (research_id,))
            research = await cursor.fetchone()
            
            if not research:
                return None
            
            # Получаем источники
            cursor = await db.execute('''
                SELECT * FROM research_sources WHERE research_id = ?
            ''', (research_id,))
            sources = await cursor.fetchall()
            
            # Получаем результаты анализа
            cursor = await db.execute('''
                SELECT * FROM research_analysis 
                WHERE research_id = ? AND analysis_type = 'general'
            ''', (research_id,))
            analysis = await cursor.fetchone()
            
            return {
                "id": research_id,
                "query": research['query'],
                "created_at": research['created_at'],
                "summary": research['summary'],
                "sources": [dict(s) for s in sources],
                "analysis": json.loads(analysis['analysis_data']) if analysis else None
            }

    async def get_user_researches(self, user_id: str) -> List[Dict[str, Any]]:
        """Получение списка исследований пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute('''
                SELECT id, query, created_at, summary 
                FROM researches 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            
            researches = await cursor.fetchall()
            return [dict(r) for r in researches]

    async def search_researches(self, query: str) -> List[Dict[str, Any]]:
        """Поиск по существующим исследованиям"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # SQLite не поддерживает ILIKE, используем LIKE с lower()
            search_pattern = f"%{query.lower()}%"
            cursor = await db.execute('''
                SELECT DISTINCT r.* 
                FROM researches r
                LEFT JOIN research_sources s ON r.id = s.research_id
                WHERE 
                    lower(r.query) LIKE ? 
                    OR lower(r.summary) LIKE ?
                    OR lower(s.content) LIKE ?
                ORDER BY r.created_at DESC
                LIMIT 10
            ''', (search_pattern, search_pattern, search_pattern))
            
            researches = await cursor.fetchall()
            return [dict(r) for r in researches]

    async def create_research_relation(
        self, 
        source_id: str, 
        target_id: str, 
        relation_type: str,
        similarity_score: float = None,
        metadata: Dict = None
    ) -> str:
        """Создание связи между исследованиями"""
        relation_id = str(uuid.uuid4())
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO research_relations (
                    id, source_research_id, target_research_id, 
                    relation_type, similarity_score, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                relation_id, source_id, target_id, relation_type,
                similarity_score, json.dumps(metadata) if metadata else None
            ))
            await db.commit()
        
        return relation_id

    async def get_research_relations(self, research_id: str) -> List[Dict[str, Any]]:
        """Получение всех связей исследования"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Получаем связи, где исследование является источником или целью
            cursor = await db.execute('''
                SELECT r.*, 
                       s1.query as source_query,
                       s2.query as target_query
                FROM research_relations r
                JOIN researches s1 ON r.source_research_id = s1.id
                JOIN researches s2 ON r.target_research_id = s2.id
                WHERE source_research_id = ? OR target_research_id = ?
                ORDER BY r.created_at DESC
            ''', (research_id, research_id))
            
            relations = await cursor.fetchall()
            return [
                {
                    **dict(r),
                    'metadata': json.loads(r['metadata']) if r['metadata'] else None
                }
                for r in relations
            ]

    async def find_similar_researches(
        self, 
        research_id: str, 
        min_similarity: float = 0.5,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Поиск похожих исследований по similarity_score"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute('''
                SELECT r.*, s2.query as similar_query, 
                       rel.similarity_score, rel.relation_type
                FROM research_relations rel
                JOIN researches s2 ON 
                    CASE 
                        WHEN rel.source_research_id = ? 
                        THEN rel.target_research_id = s2.id
                        ELSE rel.source_research_id = s2.id
                    END
                WHERE (rel.source_research_id = ? OR rel.target_research_id = ?)
                AND rel.similarity_score >= ?
                ORDER BY rel.similarity_score DESC
                LIMIT ?
            ''', (research_id, research_id, research_id, min_similarity, limit))
            
            similar = await cursor.fetchall()
            return [dict(r) for r in similar]

    async def _calculate_similarity(
        self, 
        query1: str, 
        summary1: str, 
        query2: str, 
        summary2: str
    ) -> float:
        """Вычисление схожести между двумя исследованиями"""
        try:
            # Используем TF-IDF для вычисления схожести
            vectorizer = TfidfVectorizer()
            texts = [f"{query1} {summary1}", f"{query2} {summary2}"]
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Вычисляем косинусное сходство
            similarity = (tfidf_matrix * tfidf_matrix.T).A[0][1]
            return float(similarity)
        except Exception:
            return 0.0

    async def _find_common_topics(self, summary1: str, summary2: str) -> List[str]:
        """Поиск общих тем между двумя исследованиями"""
        try:
            # Извлекаем ключевые слова из обоих текстов
            vectorizer = TfidfVectorizer(max_features=10)
            texts = [summary1, summary2]
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Получаем общие важные слова
            feature_names = vectorizer.get_feature_names_out()
            scores1 = tfidf_matrix[0].toarray()[0]
            scores2 = tfidf_matrix[1].toarray()[0]
            
            common_topics = []
            for word, score1, score2 in zip(feature_names, scores1, scores2):
                if score1 > 0 and score2 > 0:
                    common_topics.append(word)
            
            return common_topics[:5]  # Возвращаем топ-5 общих тем
        except Exception:
            return []

class TaskPriorityManager:
    """Менеджер приоритетов задач"""
    
    # Пороги для повышения приоритета (в минутах до дедлайна)
    CRITICAL_THRESHOLD = 15
    HIGH_THRESHOLD = 60
    MEDIUM_THRESHOLD = 180
    
    @classmethod
    def calculate_priority(cls, task: Task) -> TaskPriority:
        """Вычисление приоритета задачи на основе дедлайна"""
        if not task.deadline:
            return task.priority
            
        time_left = task.deadline - datetime.now()
        minutes_left = time_left.total_seconds() / 60
        
        # Определяем базовый приоритет на основе времени до дедлайна
        if minutes_left <= cls.CRITICAL_THRESHOLD:
            base_priority = TaskPriority.CRITICAL
        elif minutes_left <= cls.HIGH_THRESHOLD:
            base_priority = TaskPriority.HIGH
        elif minutes_left <= cls.MEDIUM_THRESHOLD:
            base_priority = TaskPriority.MEDIUM
        else:
            base_priority = task.priority
            
        # Никогда не понижаем приоритет ниже изначального
        return min(base_priority, task.priority)
    
    @classmethod
    def should_update_priority(cls, task: Task) -> bool:
        """Проверка необходимости обновления приоритета"""
        if not task.deadline:
            return False
            
        new_priority = cls.calculate_priority(task)
        return new_priority.value < task.priority.value

class Task:
    def __init__(self, id: str, type: str, priority: TaskPriority, parameters: Dict, deadline: Optional[datetime] = None, dependencies: List[str] = None):
        self.id = id
        self.type = type
        self.priority = priority
        self.parameters = parameters
        self.deadline = deadline
        self.dependencies = dependencies or []
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.result = None
    
    def __lt__(self, other):
        # Для поддержки сравнения в PriorityQueue
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        if self.deadline:
            return True
        return False

class TaskNotificationManager:
    """Менеджер уведомлений о задачах"""
    
    def __init__(self):
        self.notification_queue = Queue()
        self.notification_history = []
        self.notification_settings = {
            TaskStatus.COMPLETED: True,  # Уведомлять о завершении
            TaskStatus.FAILED: True,     # Уведомлять о неудачах
            TaskStatus.BLOCKED: True,    # Уведомлять о блокировке
            TaskStatus.IN_PROGRESS: False # Не уведомлять о начале выполнения
        }
        self.priority_thresholds = {
            TaskPriority.CRITICAL: 15,   # За 15 минут до дедлайна
            TaskPriority.HIGH: 60,       # За час до дедлайна
            TaskPriority.MEDIUM: 180     # За 3 часа до дедлайна
        }
    
    def create_notification(self, task: Task, event_type: str, details: Dict = None) -> Dict:
        """Создание уведомления о задаче"""
        notification = {
            "id": str(uuid.uuid4()),
            "task_id": task.id,
            "task_type": task.type,
            "priority": task.priority.name,
            "event_type": event_type,
            "details": details or {},
            "created_at": datetime.now().isoformat(),
            "is_read": False
        }
        
        if task.deadline:
            notification["deadline"] = task.deadline.isoformat()
            time_left = task.deadline - datetime.now()
            notification["minutes_until_deadline"] = time_left.total_seconds() / 60
        
        self.notification_queue.put(notification)
        self.notification_history.append(notification)
        return notification
    
    def create_deadline_notification(self, task: Task) -> Optional[Dict]:
        """Создание уведомления о приближающемся дедлайне"""
        if not task.deadline:
            return None
            
        time_left = task.deadline - datetime.now()
        minutes_left = time_left.total_seconds() / 60
        
        # Проверяем, нужно ли отправлять уведомление
        threshold = self.priority_thresholds.get(task.priority)
        if threshold and minutes_left <= threshold:
            return self.create_notification(
                task,
                "deadline_approaching",
                {
                    "minutes_left": minutes_left,
                    "threshold": threshold,
                    "urgency": "high" if minutes_left <= threshold / 2 else "medium"
                }
            )
        return None
    
    def create_status_change_notification(self, task: Task, old_status: TaskStatus) -> Optional[Dict]:
        """Создание уведомления об изменении статуса задачи"""
        if not self.notification_settings.get(task.status, False):
            return None
            
        return self.create_notification(
            task,
            "status_changed",
            {
                "old_status": old_status.value,
                "new_status": task.status.value,
                "time_in_previous_status": (
                    datetime.now() - (task.started_at or task.created_at)
                ).total_seconds() / 60
            }
        )
    
    def create_priority_change_notification(self, task: Task, old_priority: TaskPriority) -> Dict:
        """Создание уведомления об изменении приоритета задачи"""
        return self.create_notification(
            task,
            "priority_changed",
            {
                "old_priority": old_priority.name,
                "new_priority": task.priority.name,
                "reason": "deadline_approaching" if task.deadline else "manual_change"
            }
        )
    
    def get_pending_notifications(self, limit: int = 10) -> List[Dict]:
        """Получение списка непрочитанных уведомлений"""
        notifications = []
        while not self.notification_queue.empty() and len(notifications) < limit:
            notifications.append(self.notification_queue.get())
        return notifications
    
    def mark_as_read(self, notification_ids: List[str]):
        """Отметить уведомления как прочитанные"""
        for notification in self.notification_history:
            if notification["id"] in notification_ids:
                notification["is_read"] = True
    
    def get_notification_history(
        self,
        task_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Получение истории уведомлений с фильтрацией"""
        filtered_history = self.notification_history
        
        if task_id:
            filtered_history = [n for n in filtered_history if n["task_id"] == task_id]
        
        if event_types:
            filtered_history = [n for n in filtered_history if n["event_type"] in event_types]
        
        if start_date:
            filtered_history = [
                n for n in filtered_history 
                if datetime.fromisoformat(n["created_at"]) >= start_date
            ]
        
        if end_date:
            filtered_history = [
                n for n in filtered_history 
                if datetime.fromisoformat(n["created_at"]) <= end_date
            ]
        
        return sorted(
            filtered_history,
            key=lambda x: x["created_at"],
            reverse=True
        )[:limit]
    
    def get_notification_summary(self) -> Dict:
        """Получение сводки по уведомлениям"""
        unread_count = len([n for n in self.notification_history if not n["is_read"]])
        event_types = Counter(n["event_type"] for n in self.notification_history)
        priorities = Counter(n["priority"] for n in self.notification_history)
        
        return {
            "total_notifications": len(self.notification_history),
            "unread_notifications": unread_count,
            "event_type_distribution": dict(event_types),
            "priority_distribution": dict(priorities),
            "latest_notification": (
                self.notification_history[-1] 
                if self.notification_history 
                else None
            )
        }

class TaskQueue:
    def __init__(self):
        self.queue = PriorityQueue()
        self.tasks = {}  # id -> Task
        self.blocked_tasks = set()
        self.dependency_graph = {}  # id -> List[id]
        self.reverse_dependencies = {}  # id -> List[id]
        self.priority_manager = TaskPriorityManager()
        self._last_priority_check = datetime.now()
        self._priority_check_interval = timedelta(minutes=5)
        self.notification_manager = TaskNotificationManager()
    
    def add_task(self, task: Task):
        """Добавление задачи в очередь"""
        self.tasks[task.id] = task
        
        # Создаем уведомление о новой задаче
        self.notification_manager.create_notification(
            task,
            "task_created",
            {
                "has_deadline": bool(task.deadline),
                "has_dependencies": bool(task.dependencies)
            }
        )
        
        # Проверяем дедлайн
        if task.deadline:
            self.notification_manager.create_deadline_notification(task)
        
        # Обновляем граф зависимостей
        self.dependency_graph[task.id] = task.dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.reverse_dependencies:
                self.reverse_dependencies[dep_id] = []
            self.reverse_dependencies[dep_id].append(task.id)
        
        # Проверяем, заблокирована ли задача
        if any(dep_id not in self.tasks or 
               self.tasks[dep_id].status != TaskStatus.COMPLETED 
               for dep_id in task.dependencies):
            self.blocked_tasks.add(task.id)
            # Создаем уведомление о блокировке
            self.notification_manager.create_notification(
                task,
                "task_blocked",
                {"blocking_dependencies": [
                    dep_id for dep_id in task.dependencies
                    if dep_id not in self.tasks or
                    self.tasks[dep_id].status != TaskStatus.COMPLETED
                ]}
            )
        else:
            self.queue.put(task)
    
    def _check_priorities(self):
        """Проверка и обновление приоритетов задач"""
        current_time = datetime.now()
        
        if current_time - self._last_priority_check < self._priority_check_interval:
            return
            
        self._last_priority_check = current_time
        updated_tasks = []
        new_queue = PriorityQueue()
        
        while not self.queue.empty():
            task = self.queue.get()
            
            # Проверяем необходимость обновления приоритета
            if TaskPriorityManager.should_update_priority(task):
                old_priority = task.priority
                task.priority = TaskPriorityManager.calculate_priority(task)
                
                # Создаем уведомление об изменении приоритета
                self.notification_manager.create_priority_change_notification(
                    task, old_priority
                )
                
                updated_tasks.append({
                    "task_id": task.id,
                    "old_priority": old_priority.name,
                    "new_priority": task.priority.name,
                    "deadline": task.deadline.isoformat() if task.deadline else None
                })
            
            # Проверяем необходимость уведомления о дедлайне
            if task.deadline:
                self.notification_manager.create_deadline_notification(task)
            
            new_queue.put(task)
        
        self.queue = new_queue
        
        if updated_tasks:
            print(f"Updated priorities for {len(updated_tasks)} tasks: {json.dumps(updated_tasks, indent=2)}")
    
    def get_notifications(self, limit: int = 10) -> List[Dict]:
        """Получение последних уведомлений"""
        return self.notification_manager.get_pending_notifications(limit)
    
    def get_notification_history(self, **kwargs) -> List[Dict]:
        """Получение истории уведомлений"""
        return self.notification_manager.get_notification_history(**kwargs)
    
    def get_notification_summary(self) -> Dict:
        """Получение сводки по уведомлениям"""
        return self.notification_manager.get_notification_summary()

class ActionExecutor:
    """Класс для выполнения действий с приоритизацией"""
    def __init__(self):
        self.action_history = []
        self.current_action = None
        self.action_results = {}
        self.task_queue = TaskQueue()
    
    def create_task(
        self,
        action_type: str,
        parameters: Dict,
        priority: TaskPriority = TaskPriority.MEDIUM,
        deadline: Optional[datetime] = None,
        dependencies: List[str] = None,
        estimated_duration: Optional[int] = None  # в минутах
    ) -> str:
        """Создание новой задачи"""
        # Если указана ожидаемая длительность, автоматически устанавливаем дедлайн
        if estimated_duration and not deadline:
            deadline = datetime.now() + timedelta(minutes=estimated_duration * 1.5)  # +50% запаса
        
        task = Task(
            id=str(uuid.uuid4()),
            type=action_type,
            priority=priority,
            parameters=parameters,
            deadline=deadline,
            dependencies=dependencies
        )
        self.task_queue.add_task(task)
        return task.id
    
    async def process_next_task(self) -> Optional[Dict]:
        """Обработка следующей задачи из очереди"""
        task = self.task_queue.get_next_task()
        if not task:
            return None
        
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            
            # Выполняем действие
            result = await self._execute_task(task)
            
            # Отмечаем задачу как выполненную
            self.task_queue.complete_task(task.id, result)
            
            return {
                "task_id": task.id,
                "status": "completed",
                "result": result
            }
        except Exception as e:
            self.task_queue.fail_task(task.id, str(e))
            return {
                "task_id": task.id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_task(self, task: Task) -> Any:
        """Выполнение конкретной задачи"""
        # Здесь будет логика выполнения различных типов задач
        # Пока возвращаем заглушку
        return {
            "type": task.type,
            "parameters": task.parameters,
            "executed_at": datetime.now().isoformat()
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Получение статуса задачи"""
        return self.task_queue.get_task_status(task_id)
    
    def get_queue_status(self) -> Dict:
        """Получение статуса очереди задач"""
        return self.task_queue.get_queue_status()
    
    def cancel_task(self, task_id: str):
        """Отмена задачи"""
        self.task_queue.cancel_task(task_id)

class MultiAgentSystem:
    def __init__(self, openai_api_key: str):
        self.messages_queue = Queue()
        self.openai_api_key = openai_api_key
        self.agents = {}
        self.current_plan = None
        self.current_research = None
        self.nlp_analyzer = NLPAnalyzer()
        
        # Инициализация хранилища исследований с SQLite
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'research.db')
        self.research_storage = ResearchStorage(db_path)
        
        self.action_executor = ActionExecutor()
        self._setup_agents()

    async def initialize(self):
        """Инициализация системы"""
        await self.research_storage.initialize()

    def _setup_agents(self):
        # Исследователь
        researcher = self._create_researcher_agent()
        # Планировщик
        planner = self._create_planner_agent()
        # Исполнитель (основной агент)
        executor = self._create_executor_agent()

        self.agents = {
            AgentRole.RESEARCHER: researcher,
            AgentRole.PLANNER: planner,
            AgentRole.EXECUTOR: executor
        }

    def _create_base_llm(self, temperature: float = 0):
        return ChatOpenAI(
            temperature=temperature,
            model="gpt-4-1106-preview",
            api_key=self.openai_api_key
        )

    def _create_researcher_agent(self):
        tools = [
            Tool(
                name="WebSearch",
                func=self._web_search,
                description="Поиск информации в интернете через DuckDuckGo"
            ),
            Tool(
                name="ContentExtractor",
                func=self._extract_content,
                description="Извлечение содержимого веб-страницы по URL"
            ),
            Tool(
                name="AdvancedTextAnalysis",
                func=self._advanced_text_analysis,
                description="Продвинутый анализ текста с использованием NLP"
            ),
            Tool(
                name="SearchPreviousResearch",
                func=self._search_previous_research,
                description="Поиск по предыдущим исследованиям"
            ),
            Tool(
                name="GetResearchHistory",
                func=self._get_research_history,
                description="Получение истории исследований пользователя"
            ),
            Tool(
                name="GetRelatedResearches",
                func=self._get_related_researches,
                description="Получение связанных исследований"
            ),
            Tool(
                name="FindSimilarResearches",
                func=self._find_similar_researches,
                description="Поиск похожих исследований по схожести"
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты - агент-исследователь с продвинутыми аналитическими способностями. "
                      "Твои задачи:\n"
                      "1. Собирать информацию из различных источников\n"
                      "2. Проводить глубокий анализ текста с использованием NLP\n"
                      "3. Выявлять тональность и ключевые темы\n"
                      "4. Извлекать важные фразы и концепции\n"
                      "5. Создавать подробные аналитические отчеты\n"
                      "Используй все доступные инструменты для глубокого исследования."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        return self._create_agent_executor(tools, prompt, temperature=0.3)

    def _create_planner_agent(self):
        tools = [
            Tool(
                name="CreateStructuredPlan",
                func=self._create_structured_plan,
                description="Создание структурированного плана действий в формате JSON"
            ),
            Tool(
                name="UpdatePlan",
                func=self._update_plan,
                description="Обновление существующего плана новыми шагами или изменение статуса"
            ),
            Tool(
                name="EvaluateAndOptimizePlan",
                func=self._evaluate_and_optimize_plan,
                description="Оценка и оптимизация плана с учетом ограничений и возможностей"
            ),
            Tool(
                name="GetPlanStatus",
                func=self._get_plan_status,
                description="Получение текущего статуса плана и его шагов"
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты - агент-планировщик. Твоя задача - создавать структурированные планы действий "
                      "на основе информации от исследователя и передавать их исполнителю. "
                      "Каждый план должен быть четким, измеримым и иметь конкретные шаги. "
                      "Используй инструменты для создания и управления планами в JSON формате."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        return self._create_agent_executor(tools, prompt, temperature=0.2)

    def _create_executor_agent(self):
        tools = [
            Tool(
                name="ExecuteAction",
                func=self._execute_action,
                description="Выполнение конкретных действий из плана"
            ),
            Tool(
                name="GetActionStatus",
                func=self._get_action_status,
                description="Получение статуса выполнения действия"
            ),
            Tool(
                name="GetActionHistory",
                func=self._get_action_history,
                description="Получение истории выполненных действий"
            ),
            Tool(
                name="SendNotification",
                func=self._send_notification,
                description="Отправка уведомления пользователю"
            ),
            Tool(
                name="SaveResult",
                func=self._save_result,
                description="Сохранение результата выполнения действия"
            ),
            Tool(
                name="ValidateResult",
                func=self._validate_result,
                description="Проверка результата на соответствие требованиям"
            ),
            Tool(
                name="GenerateReport",
                func=self._generate_report,
                description="Создание отчета о выполненных действиях"
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты - агент-исполнитель с широкими возможностями. Твои задачи:\n"
                      "1. Выполнение действий из плана\n"
                      "2. Отслеживание статуса выполнения\n"
                      "3. Валидация результатов\n"
                      "4. Создание отчетов\n"
                      "5. Уведомление пользователя\n"
                      "6. Обработка ошибок и восстановление\n"
                      "Используй доступные инструменты для эффективного выполнения задач."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        return self._create_agent_executor(tools, prompt)

    def _create_agent_executor(self, tools, prompt, temperature: float = 0):
        llm = self._create_base_llm(temperature)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True
        )

    # Инструменты для агентов
    def _web_search(self, query: str) -> str:
        """Поиск информации через DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            
            if not self.current_research:
                self.current_research = ResearchResult(query)
            
            for result in results:
                self.current_research.add_source(
                    title=result.get('title', 'Без названия'),
                    content=result.get('body', ''),
                    url=result.get('link', None)
                )
            
            return self.current_research.to_json()
        except Exception as e:
            return f"Ошибка при поиске: {str(e)}"

    def _extract_content(self, url: str) -> str:
        """Извлечение содержимого веб-страницы"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Удаляем ненужные элементы
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()
            
            content = {
                "title": soup.title.string if soup.title else "Без названия",
                "text": soup.get_text(separator='\n', strip=True),
                "url": url
            }
            
            if self.current_research:
                self.current_research.add_source(**content)
            
            return json.dumps(content, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при извлечении контента: {str(e)}"

    def _analyze_data(self, data: str) -> str:
        """Продвинутый анализ данных"""
        try:
            # Преобразуем строку JSON в словарь
            data_dict = json.loads(data)
            
            # Создаем DataFrame для анализа текстовых данных
            if isinstance(data_dict, list):
                df = pd.DataFrame(data_dict)
            else:
                df = pd.DataFrame([data_dict])
            
            analysis = {
                "basic_stats": {
                    "row_count": len(df),
                    "column_count": len(df.columns)
                },
                "text_analysis": {},
                "numerical_analysis": {}
            }
            
            # Анализ текстовых полей
            text_columns = df.select_dtypes(include=['object']).columns
            for col in text_columns:
                if col in df:
                    text_data = ' '.join(df[col].astype(str))
                    analysis["text_analysis"][col] = {
                        "word_count": len(text_data.split()),
                        "unique_words": len(set(text_data.split())),
                        "avg_word_length": np.mean([len(word) for word in text_data.split()])
                    }
            
            # Анализ числовых полей
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df:
                    analysis["numerical_analysis"][col] = {
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                        "std": float(df[col].std())
                    }
            
            if self.current_research:
                self.current_research.set_analysis(analysis)
            
            return json.dumps(analysis, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при анализе данных: {str(e)}"

    def _summarize_findings(self, research_data: str = None) -> str:
        """Создание краткого содержания исследования"""
        try:
            if not self.current_research:
                return "Нет текущего исследования для обобщения"
            
            # Собираем все тексты из источников
            all_texts = [
                f"Источник: {source['title']}\n{source['content']}"
                for source in self.current_research.sources
            ]
            
            # Создаем краткое содержание
            summary = {
                "query": self.current_research.query,
                "sources_count": len(self.current_research.sources),
                "key_findings": self._extract_key_findings(all_texts),
                "main_topics": self._extract_main_topics(all_texts)
            }
            
            self.current_research.set_summary(json.dumps(summary, ensure_ascii=False, indent=2))
            return self.current_research.to_json()
        except Exception as e:
            return f"Ошибка при создании краткого содержания: {str(e)}"

    def _compare_information(self, sources: str) -> str:
        """Сравнение информации из разных источников"""
        try:
            if not self.current_research or not self.current_research.sources:
                return "Нет источников для сравнения"
            
            comparison = {
                "similarities": [],
                "differences": [],
                "confidence_score": {}
            }
            
            # Здесь можно добавить более сложную логику сравнения
            # Например, использование NLP для выявления противоречий
            
            return json.dumps(comparison, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при сравнении информации: {str(e)}"

    def _fact_check(self, statement: str) -> str:
        """Проверка достоверности информации"""
        try:
            # Здесь можно добавить интеграцию с сервисами проверки фактов
            check_result = {
                "statement": statement,
                "confidence": 0.0,
                "supporting_evidence": [],
                "contradicting_evidence": []
            }
            
            return json.dumps(check_result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при проверке фактов: {str(e)}"

    def _extract_key_findings(self, texts: List[str]) -> List[str]:
        """Извлечение ключевых находок из текстов"""
        # Здесь можно добавить более сложную логику извлечения
        return ["Ключевая находка 1", "Ключевая находка 2"]

    def _extract_main_topics(self, texts: List[str]) -> List[str]:
        """Извлечение основных тем из текстов"""
        # Здесь можно добавить более сложную логику извлечения тем
        return ["Тема 1", "Тема 2"]

    def _create_structured_plan(self, goal: str, context: str = None) -> str:
        """Создание нового структурированного плана"""
        try:
            self.current_plan = ActionPlan(goal, context)
            
            # Разбиваем цель на конкретные шаги (это пример, можно расширить логику)
            steps = [
                ("analyze", "Анализ полученной информации", 5),
                ("prepare", "Подготовка необходимых ресурсов", 10),
                ("execute", "Выполнение основной задачи", 15),
                ("verify", "Проверка результатов", 5)
            ]
            
            for action, desc, time in steps:
                self.current_plan.add_step(action, desc, time)
            
            return self.current_plan.to_json()
        except Exception as e:
            return f"Ошибка при создании плана: {str(e)}"

    def _update_plan(self, plan_update: str) -> str:
        """Обновление существующего плана"""
        try:
            if not self.current_plan:
                return "План не найден. Сначала создайте план."
            
            update_data = json.loads(plan_update)
            
            if "new_steps" in update_data:
                for step in update_data["new_steps"]:
                    self.current_plan.add_step(
                        step["action"],
                        step["description"],
                        step.get("estimated_time")
                    )
            
            if "status_updates" in update_data:
                for i, status in update_data["status_updates"].items():
                    if 0 <= int(i) < len(self.current_plan.steps):
                        self.current_plan.steps[int(i)].status = status
            
            return self.current_plan.to_json()
        except Exception as e:
            return f"Ошибка при обновлении плана: {str(e)}"

    def _evaluate_and_optimize_plan(self, constraints: str = None) -> str:
        """Оценка и оптимизация текущего плана"""
        try:
            if not self.current_plan:
                return "План не найден. Сначала создайте план."
            
            # Анализ плана
            total_time = sum(step.estimated_time or 0 for step in self.current_plan.steps)
            pending_steps = sum(1 for step in self.current_plan.steps if step.status == "pending")
            
            analysis = {
                "total_steps": len(self.current_plan.steps),
                "pending_steps": pending_steps,
                "completed_steps": len(self.current_plan.steps) - pending_steps,
                "estimated_total_time": total_time,
                "optimization_suggestions": []
            }
            
            # Добавляем предложения по оптимизации
            if total_time > 60:
                analysis["optimization_suggestions"].append(
                    "План может быть слишком длительным. Рекомендуется разбить на подзадачи."
                )
            
            if len(self.current_plan.steps) > 10:
                analysis["optimization_suggestions"].append(
                    "Большое количество шагов. Рекомендуется объединить похожие задачи."
                )
            
            return json.dumps(analysis, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при оценке плана: {str(e)}"

    def _get_plan_status(self) -> str:
        """Получение текущего статуса плана"""
        try:
            if not self.current_plan:
                return "План не найден. Сначала создайте план."
            
            status = {
                "goal": self.current_plan.goal,
                "status": self.current_plan.status,
                "steps_status": [
                    {
                        "action": step.action,
                        "status": step.status,
                        "estimated_time": step.estimated_time
                    }
                    for step in self.current_plan.steps
                ]
            }
            
            return json.dumps(status, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при получении статуса плана: {str(e)}"

    def _execute_action(self, action: str) -> str:
        # Заглушка для демонстрации
        return f"Выполнено действие: {action}"

    def _calculator(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Ошибка в вычислении: {str(e)}"

    async def process_message(self, message: str, user_id: str = None) -> str:
        """Обработка входящего сообщения с использованием системы агентов"""
        try:
            # 1. Поиск похожих исследований
            previous_research = await self._search_previous_research(message)
            
            # 2. Исследователь собирает информацию
            research_result = await self.agents[AgentRole.RESEARCHER].arun(
                f"Исследуй следующий запрос: {message}\n"
                f"Предыдущие исследования: {previous_research}"
            )
            
            # 3. Сохраняем результаты исследования
            if self.current_research:
                research_id = await self.research_storage.save_research(
                    self.current_research, 
                    user_id
                )
            
            # 4. Планировщик создает план на основе исследования
            plan = await self.agents[AgentRole.PLANNER].arun(
                f"Создай план на основе исследования: {research_result}"
            )
            
            # 5. Исполнитель выполняет план и взаимодействует с пользователем
            final_response = await self.agents[AgentRole.EXECUTOR].arun(
                f"Выполни план и подготовь ответ пользователю: {plan}"
            )
            
            return final_response
        except Exception as e:
            return f"Произошла ошибка при обработке запроса: {str(e)}"

    def _advanced_text_analysis(self, text: str) -> str:
        """Продвинутый анализ текста с использованием NLP"""
        try:
            analysis_result = self.nlp_analyzer.analyze_text(text)
            return json.dumps(analysis_result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при анализе текста: {str(e)}"

    def _analyze_sentiment_standalone(self, text: str) -> str:
        """Отдельный инструмент для анализа тональности"""
        try:
            sentiment_result = self.nlp_analyzer._analyze_sentiment(text)
            return json.dumps(sentiment_result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при анализе тональности: {str(e)}"

    def _analyze_topics_standalone(self, text: str) -> str:
        """Отдельный инструмент для анализа тем"""
        try:
            topics_result = self.nlp_analyzer._analyze_topics(text)
            return json.dumps(topics_result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при анализе тем: {str(e)}"

    def _extract_key_phrases_standalone(self, text: str) -> str:
        """Отдельный инструмент для извлечения ключевых фраз"""
        try:
            doc = self.nlp_analyzer.nlp(text)
            phrases_result = self.nlp_analyzer._extract_key_phrases(doc)
            return json.dumps(phrases_result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при извлечении ключевых фраз: {str(e)}"

    async def _search_previous_research(self, query: str) -> str:
        """Поиск по предыдущим исследованиям"""
        try:
            results = await self.research_storage.search_researches(query)
            return json.dumps({
                "found_researches": len(results),
                "results": results
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при поиске исследований: {str(e)}"

    async def _get_research_history(self, user_id: str) -> str:
        """Получение истории исследований пользователя"""
        try:
            history = await self.research_storage.get_user_researches(user_id)
            return json.dumps({
                "total_researches": len(history),
                "history": history
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при получении истории исследований: {str(e)}"

    async def _get_related_researches(self, research_id: str) -> str:
        """Получение связанных исследований"""
        try:
            relations = await self.research_storage.get_research_relations(research_id)
            return json.dumps({
                "total_relations": len(relations),
                "relations": relations
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при получении связанных исследований: {str(e)}"

    async def _find_similar_researches(self, research_id: str) -> str:
        """Поиск похожих исследований"""
        try:
            similar = await self.research_storage.find_similar_researches(
                research_id,
                min_similarity=0.5,
                limit=5
            )
            return json.dumps({
                "total_similar": len(similar),
                "similar_researches": similar
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при поиске похожих исследований: {str(e)}"

    async def _execute_action(self, action_data: str) -> str:
        """Выполнение действия из плана"""
        try:
            action = json.loads(action_data)
            action_id = str(uuid.uuid4())
            
            # Начинаем выполнение действия
            self.action_executor.start_action(
                action_id,
                action.get("type", "unknown"),
                action.get("parameters")
            )
            
            # Выполняем действие в зависимости от типа
            result = await self._process_action(action)
            
            # Завершаем действие
            self.action_executor.complete_action(result)
            
            return json.dumps({
                "action_id": action_id,
                "status": "completed",
                "result": result
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.action_executor.current_action:
                self.action_executor.fail_action(str(e))
            return f"Ошибка при выполнении действия: {str(e)}"

    async def _process_action(self, action: Dict) -> Any:
        """Обработка различных типов действий"""
        action_type = action.get("type", "").lower()
        parameters = action.get("parameters", {})
        
        if action_type == "research":
            return await self._process_research_action(parameters)
        elif action_type == "analysis":
            return await self._process_analysis_action(parameters)
        elif action_type == "notification":
            return await self._process_notification_action(parameters)
        elif action_type == "report":
            return await self._process_report_action(parameters)
        else:
            raise ValueError(f"Неизвестный тип действия: {action_type}")

    async def _process_research_action(self, parameters: Dict) -> Dict:
        """Обработка исследовательского действия"""
        query = parameters.get("query")
        if not query:
            raise ValueError("Не указан запрос для исследования")
        
        research_result = await self.research_storage.search_researches(query)
        return {
            "type": "research",
            "query": query,
            "results": research_result
        }

    async def _process_analysis_action(self, parameters: Dict) -> Dict:
        """Обработка аналитического действия"""
        text = parameters.get("text")
        if not text:
            raise ValueError("Не указан текст для анализа")
        
        analysis_result = self.nlp_analyzer.analyze_text(text)
        return {
            "type": "analysis",
            "text": text,
            "results": analysis_result
        }

    async def _process_notification_action(self, parameters: Dict) -> Dict:
        """Обработка действия уведомления"""
        message = parameters.get("message")
        if not message:
            raise ValueError("Не указано сообщение для уведомления")
        
        # Здесь можно добавить реальную отправку уведомлений
        return {
            "type": "notification",
            "message": message,
            "sent_at": datetime.now().isoformat()
        }

    async def _process_report_action(self, parameters: Dict) -> Dict:
        """Обработка действия создания отчета"""
        report_type = parameters.get("type", "general")
        data = parameters.get("data", {})
        
        report = await self._generate_report(json.dumps({
            "type": report_type,
            "data": data
        }))
        
        return {
            "type": "report",
            "report_type": report_type,
            "report": report
        }

    async def _get_action_status(self, action_id: str) -> str:
        """Получение статуса выполнения действия"""
        try:
            history = self.action_executor.get_action_history()
            action = next((a for a in history if a["id"] == action_id), None)
            
            if not action:
                return "Действие не найдено"
            
            return json.dumps(action, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при получении статуса: {str(e)}"

    async def _get_action_history(self) -> str:
        """Получение истории выполненных действий"""
        try:
            history = self.action_executor.get_action_history()
            return json.dumps({
                "total_actions": len(history),
                "history": history
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при получении истории: {str(e)}"

    async def _send_notification(self, message: str) -> str:
        """Отправка уведомления пользователю"""
        try:
            # Здесь можно добавить реальную отправку уведомлений
            notification = {
                "message": message,
                "sent_at": datetime.now().isoformat(),
                "status": "sent"
            }
            return json.dumps(notification, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при отправке уведомления: {str(e)}"

    async def _save_result(self, result_data: str) -> str:
        """Сохранение результата выполнения действия"""
        try:
            result = json.loads(result_data)
            # Здесь можно добавить сохранение в базу данных
            return json.dumps({
                "status": "saved",
                "result_id": str(uuid.uuid4()),
                "saved_at": datetime.now().isoformat()
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при сохранении результата: {str(e)}"

    async def _validate_result(self, result_data: str) -> str:
        """Проверка результата на соответствие требованиям"""
        try:
            result = json.loads(result_data)
            # Здесь можно добавить реальную валидацию
            validation = {
                "is_valid": True,
                "checks": [
                    {"name": "format", "passed": True},
                    {"name": "completeness", "passed": True}
                ],
                "validated_at": datetime.now().isoformat()
            }
            return json.dumps(validation, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при валидации: {str(e)}"

    async def _generate_report(self, report_data: str) -> str:
        """Создание отчета о выполненных действиях"""
        try:
            data = json.loads(report_data)
            report = {
                "type": data.get("type", "general"),
                "generated_at": datetime.now().isoformat(),
                "content": data.get("data", {}),
                "summary": "Отчет успешно создан"
            }
            return json.dumps(report, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Ошибка при создании отчета: {str(e)}"

class AIAgentPlugin:
    """Плагин для работы с AI агентами"""
    
    def __init__(self, openai_api_key: str):
        self.multi_agent_system = MultiAgentSystem(openai_api_key)
    
    async def initialize(self):
        """Инициализация плагина"""
        await self.multi_agent_system.initialize()
    
    async def handle_message(self, message: str, user_id: str = None) -> str:
        """Обработка входящего сообщения от пользователя"""
        return await self.multi_agent_system.process_message(message, user_id)

    @property
    def help_message(self) -> str:
        return (
            "🤖 Multi-Agent AI System\n\n"
            "Эта система использует трех специализированных агентов:\n"
            "🔍 Исследователь - собирает и анализирует информацию\n"
            "📋 Планировщик - создает планы действий\n"
            "⚡️ Исполнитель - выполняет задачи и взаимодействует с пользователем\n\n"
            "Система запоминает все исследования и может использовать их в будущем!\n\n"
            "Просто напишите свой запрос, и система агентов поможет вам!"
        ) 