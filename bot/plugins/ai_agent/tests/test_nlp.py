"""
Тесты для NLP анализатора
"""

import pytest
import torch
from unittest.mock import Mock, patch
from ..nlp import NLPAnalyzer
from ..logging import AgentLogger


@pytest.fixture
def logger():
    """Фикстура для логгера"""
    return AgentLogger("test_logs")


@pytest.fixture
def analyzer(logger):
    """Фикстура для анализатора"""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("transformers.AutoModel.from_pretrained") as mock_model, \
         patch("transformers.pipeline") as mock_pipeline:
        
        # Настраиваем мок токенизатора
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.tokenize = Mock(
            return_value=["test", "tokens"]
        )
        mock_tokenizer.return_value.__call__ = Mock(
            return_value=Mock(
                to=Mock(return_value=Mock())
            )
        )
        
        # Настраиваем мок модели
        mock_model.return_value = Mock()
        mock_model.return_value.to = Mock(return_value=Mock())
        mock_model.return_value.last_hidden_state = torch.randn(1, 2, 768)
        
        # Настраиваем мок пайплайнов
        mock_pipeline.return_value = Mock(
            return_value=[{"label": "POSITIVE", "score": 0.9}]
        )
        
        analyzer = NLPAnalyzer(logger=logger)
        yield analyzer


@pytest.mark.asyncio
async def test_get_embeddings(analyzer):
    """Тест получения эмбеддингов"""
    texts = ["Test text 1", "Test text 2"]
    embeddings = analyzer.get_embeddings(texts)
    
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == 768  # Размерность BERT


@pytest.mark.asyncio
async def test_calculate_similarity(analyzer):
    """Тест расчета схожести"""
    text1 = "Test text 1"
    text2 = "Test text 2"
    
    similarity = analyzer.calculate_similarity(text1, text2)
    
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1


@pytest.mark.asyncio
async def test_analyze_sentiment(analyzer):
    """Тест анализа тональности"""
    text = "Test text"
    
    label, score = analyzer.analyze_sentiment(text)
    
    assert isinstance(label, str)
    assert isinstance(score, float)
    assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_extract_entities(analyzer):
    """Тест извлечения сущностей"""
    text = "Test text"
    
    # Настраиваем мок пайплайна NER
    analyzer.ner_pipeline.return_value = [
        {
            "entity": "PER",
            "word": "Test",
            "start": 0,
            "end": 4
        }
    ]
    
    entities = analyzer.extract_entities(text)
    
    assert isinstance(entities, list)
    assert len(entities) > 0
    assert "entity" in entities[0]
    assert "word" in entities[0]


@pytest.mark.asyncio
async def test_answer_question(analyzer):
    """Тест ответов на вопросы"""
    question = "Test question?"
    context = "Test context"
    
    # Настраиваем мок пайплайна QA
    analyzer.qa_pipeline.return_value = {
        "answer": "Test answer",
        "score": 0.9,
        "start": 0,
        "end": 11
    }
    
    answer = analyzer.answer_question(question, context)
    
    assert isinstance(answer, dict)
    assert "answer" in answer
    assert "score" in answer


@pytest.mark.asyncio
async def test_extract_keywords(analyzer):
    """Тест извлечения ключевых слов"""
    text = "Test text with some keywords"
    
    keywords = analyzer.extract_keywords(text, top_k=3)
    
    assert isinstance(keywords, list)
    assert len(keywords) <= 3
    assert all(isinstance(k, tuple) for k in keywords)
    assert all(isinstance(k[0], str) for k in keywords)
    assert all(isinstance(k[1], float) for k in keywords)


@pytest.mark.asyncio
async def test_analyze_topics(analyzer):
    """Тест анализа тем"""
    texts = ["Test text 1", "Test text 2"]
    
    topics = analyzer.analyze_topics(texts, num_topics=2)
    
    assert isinstance(topics, list)
    assert len(topics) == 2
    assert all(isinstance(t, list) for t in topics)
    assert all(isinstance(w, tuple) for t in topics for w in t)


@pytest.mark.asyncio
async def test_summarize(analyzer):
    """Тест генерации краткого содержания"""
    text = "Test text for summarization"
    
    # Настраиваем мок пайплайна
    with patch("transformers.pipeline") as mock_pipeline:
        mock_pipeline.return_value = Mock(
            return_value=[{"summary_text": "Test summary"}]
        )
        
        summary = analyzer.summarize(text)
        
        assert isinstance(summary, str)
        assert len(summary) > 0


@pytest.mark.asyncio
async def test_analyze_text(analyzer):
    """Тест комплексного анализа"""
    text = "Test text for analysis"
    
    # Настраиваем моки
    analyzer.analyze_sentiment = Mock(
        return_value=("POSITIVE", 0.9)
    )
    analyzer.extract_entities = Mock(
        return_value=[{"entity": "TEST", "word": "test"}]
    )
    analyzer.extract_keywords = Mock(
        return_value=[("test", 0.9)]
    )
    analyzer.summarize = Mock(
        return_value="Test summary"
    )
    
    result = analyzer.analyze_text(text)
    
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "entities" in result
    assert "keywords" in result
    assert "summary" in result


@pytest.mark.asyncio
async def test_error_handling(analyzer):
    """Тест обработки ошибок"""
    # Тест ошибки инициализации
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_tokenizer.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            NLPAnalyzer(logger=logger)
            
    # Тест ошибки анализа
    analyzer.sentiment_pipeline.side_effect = Exception("Test error")
    
    with pytest.raises(Exception):
        analyzer.analyze_sentiment("Test text")


@pytest.mark.asyncio
async def test_batch_processing(analyzer):
    """Тест пакетной обработки"""
    texts = ["Text 1", "Text 2", "Text 3"]
    batch_size = 2
    
    embeddings = analyzer.get_embeddings(texts, batch_size=batch_size)
    
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == 768


@pytest.mark.asyncio
async def test_device_handling(logger):
    """Тест работы с устройствами"""
    # Тест CPU
    with patch.dict("os.environ", {"USE_CUDA": "0"}):
        analyzer = NLPAnalyzer(logger=logger)
        assert analyzer.device == "cpu"
        
    # Тест CUDA
    with patch.dict("os.environ", {"USE_CUDA": "1"}), \
         patch("torch.cuda.is_available", return_value=True):
        analyzer = NLPAnalyzer(logger=logger)
        assert analyzer.device == "cuda"


@pytest.mark.asyncio
async def test_model_loading(logger):
    """Тест загрузки моделей"""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("transformers.AutoModel.from_pretrained") as mock_model:
        
        # Проверяем загрузку с правильными параметрами
        NLPAnalyzer(logger=logger)
        
        mock_tokenizer.assert_called_once_with(
            "DeepPavlov/rubert-base-cased"
        )
        mock_model.assert_called_once_with(
            "DeepPavlov/rubert-base-cased"
        ) 