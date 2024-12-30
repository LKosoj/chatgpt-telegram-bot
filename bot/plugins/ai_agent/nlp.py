"""
Модуль для NLP анализа
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    pipeline
)
import torch
from sklearn.metrics.pairwise import cosine_similarity
from .logging import AgentLogger


class NLPAnalyzer:
    """Анализатор естественного языка"""
    
    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[AgentLogger] = None
    ):
        """
        Инициализация анализатора
        
        Args:
            model_name: название модели
            device: устройство для вычислений
            logger: логгер (опционально)
        """
        self.logger = logger or AgentLogger("logs")
        self.device = device
        
        try:
            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            
            # Создаем пайплайны
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment",
                device=0 if device == "cuda" else -1
            )
            
            self.ner_pipeline = pipeline(
                "ner",
                model="DeepPavlov/bert-base-multilingual-cased-ner",
                device=0 if device == "cuda" else -1
            )
            
            self.qa_pipeline = pipeline(
                "question-answering",
                model="DeepPavlov/rubert-base-cased-squad",
                device=0 if device == "cuda" else -1
            )
            
            self.logger.info(
                "NLP analyzer initialized",
                {"device": device}
            )
            
        except Exception as e:
            self.logger.error(
                "Error initializing NLP analyzer",
                error=e
            )
            raise
            
    def get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Получение эмбеддингов текстов
        
        Args:
            texts: список текстов
            batch_size: размер батча
            
        Returns:
            np.ndarray: матрица эмбеддингов
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Токенизируем тексты
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Получаем эмбеддинги
            with torch.no_grad():
                outputs = self.model(**encoded)
                
            # Используем [CLS] токен как эмбеддинг предложения
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
        
    def calculate_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Расчет семантической схожести текстов
        
        Args:
            text1: первый текст
            text2: второй текст
            
        Returns:
            float: коэффициент схожести
        """
        # Получаем эмбеддинги
        embeddings = self.get_embeddings([text1, text2])
        
        # Считаем косинусное сходство
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        return float(similarity)
        
    def analyze_sentiment(
        self,
        text: str
    ) -> Tuple[str, float]:
        """
        Анализ тональности текста
        
        Args:
            text: текст
            
        Returns:
            Tuple[str, float]: метка тональности и уверенность
        """
        result = self.sentiment_pipeline(text)[0]
        return result["label"], result["score"]
        
    def extract_entities(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Извлечение именованных сущностей
        
        Args:
            text: текст
            
        Returns:
            List[Dict[str, Any]]: список сущностей
        """
        entities = self.ner_pipeline(text)
        
        # Группируем токены в сущности
        grouped = []
        current = None
        
        for entity in entities:
            if current is None:
                current = entity
            elif (
                entity["entity"] == current["entity"]
                and entity["start"] == current["end"]
            ):
                current["word"] += entity["word"].replace("##", "")
                current["end"] = entity["end"]
            else:
                grouped.append(current)
                current = entity
                
        if current:
            grouped.append(current)
            
        return grouped
        
    def answer_question(
        self,
        question: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Ответ на вопрос по контексту
        
        Args:
            question: вопрос
            context: контекст
            
        Returns:
            Dict[str, Any]: ответ с метаданными
        """
        return self.qa_pipeline(
            question=question,
            context=context
        )
        
    def extract_keywords(
        self,
        text: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Извлечение ключевых слов
        
        Args:
            text: текст
            top_k: количество слов
            
        Returns:
            List[Tuple[str, float]]: список слов с весами
        """
        # Токенизируем текст
        tokens = self.tokenizer.tokenize(text)
        
        # Фильтруем стоп-слова и пунктуацию
        filtered = [
            token for token in tokens
            if not re.match(r"[^\w\s]", token)
            and len(token) > 1
        ]
        
        # Получаем эмбеддинги токенов
        embeddings = []
        
        for i in range(0, len(filtered), 32):
            batch = filtered[i:i + 32]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(batch_embeddings)
            
        embeddings = np.vstack(embeddings)
        
        # Считаем важность токенов
        importance = np.mean(
            cosine_similarity(embeddings, embeddings),
            axis=1
        )
        
        # Сортируем по важности
        pairs = list(zip(filtered, importance))
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        return pairs[:top_k]
        
    def analyze_topics(
        self,
        texts: List[str],
        num_topics: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """
        Анализ тем в текстах
        
        Args:
            texts: список текстов
            num_topics: количество тем
            
        Returns:
            List[List[Tuple[str, float]]]: темы с весами слов
        """
        # Получаем эмбеддинги текстов
        embeddings = self.get_embeddings(texts)
        
        # Применяем SVD для выделения тем
        U, S, Vh = np.linalg.svd(embeddings, full_matrices=False)
        
        # Получаем веса слов для каждой темы
        topics = []
        
        for topic_idx in range(num_topics):
            # Получаем веса слов для темы
            topic = Vh[topic_idx]
            
            # Получаем топ слова
            pairs = list(zip(
                self.tokenizer.convert_ids_to_tokens(
                    range(len(topic))
                ),
                topic
            ))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Фильтруем служебные токены
            filtered = [
                (token, weight)
                for token, weight in pairs[:50]
                if not token.startswith("[")
                and not token.startswith("##")
                and len(token) > 1
            ]
            
            topics.append(filtered[:10])
            
        return topics
        
    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30
    ) -> str:
        """
        Генерация краткого содержания
        
        Args:
            text: текст
            max_length: максимальная длина
            min_length: минимальная длина
            
        Returns:
            str: краткое содержание
        """
        summarizer = pipeline(
            "summarization",
            model="IlyaGusev/mbart_ru_sum_gazeta",
            device=0 if self.device == "cuda" else -1
        )
        
        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )[0]
        
        return result["summary_text"]
        
    def analyze_text(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Комплексный анализ текста
        
        Args:
            text: текст
            
        Returns:
            Dict[str, Any]: результаты анализа
        """
        try:
            # Анализируем тональность
            sentiment, confidence = self.analyze_sentiment(text)
            
            # Извлекаем сущности
            entities = self.extract_entities(text)
            
            # Извлекаем ключевые слова
            keywords = self.extract_keywords(text)
            
            # Генерируем краткое содержание
            summary = self.summarize(text)
            
            return {
                "sentiment": {
                    "label": sentiment,
                    "confidence": confidence
                },
                "entities": entities,
                "keywords": [
                    {"word": word, "weight": float(weight)}
                    for word, weight in keywords
                ],
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(
                "Error analyzing text",
                error=e,
                extra={"text_length": len(text)}
            )
            raise 