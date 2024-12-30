"""
Модуль для конфигурации системы
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class NLPConfig:
    """Конфигурация NLP"""
    model_name: str = "DeepPavlov/rubert-base-cased"
    device: str = "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"
    batch_size: int = 32
    max_length: int = 512


@dataclass
class StorageConfig:
    """Конфигурация хранилища"""
    db_path: str = "data/ai_agent.db"
    cache_dir: str = "data/cache"
    max_cache_size: int = 1024 * 1024 * 100  # 100MB


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    log_dir: str = "logs"
    max_file_size: int = 1024 * 1024 * 10  # 10MB
    backup_count: int = 5
    log_level: str = "INFO"


@dataclass
class APIConfig:
    """Конфигурация API"""
    rate_limit_calls: int = 60
    rate_limit_period: int = 60
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1


@dataclass
class AgentConfig:
    """Конфигурация агента"""
    max_tasks: int = 100
    task_timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5


@dataclass
class Config:
    """Основная конфигурация"""
    nlp: NLPConfig = field(default_factory=NLPConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    debug: bool = False
    
    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """
        Загрузка конфигурации из файла
        
        Args:
            config_path: путь к файлу
            
        Returns:
            Config: конфигурация
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            return cls()
            
        with open(config_path) as f:
            data = json.load(f)
            
        return cls(
            nlp=NLPConfig(**data.get("nlp", {})),
            storage=StorageConfig(**data.get("storage", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            api=APIConfig(**data.get("api", {})),
            agent=AgentConfig(**data.get("agent", {})),
            debug=data.get("debug", False)
        )
        
    def save(self, config_path: str):
        """
        Сохранение конфигурации в файл
        
        Args:
            config_path: путь к файлу
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(
                asdict(self),
                f,
                indent=2,
                ensure_ascii=False
            )
            
    def update(self, **kwargs):
        """
        Обновление конфигурации
        
        Args:
            **kwargs: параметры
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    current = getattr(self, key)
                    for k, v in value.items():
                        if hasattr(current, k):
                            setattr(current, k, v)
                else:
                    setattr(self, key, value)


class ConfigManager:
    """Менеджер конфигурации"""
    
    def __init__(
        self,
        config_dir: str = "config",
        config_name: str = "ai_agent.json"
    ):
        """
        Инициализация менеджера
        
        Args:
            config_dir: директория конфигурации
            config_name: имя файла конфигурации
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / config_name
        self.config = Config.load(self.config_path)
        
    def get_config(self) -> Config:
        """
        Получение конфигурации
        
        Returns:
            Config: конфигурация
        """
        return self.config
        
    def update_config(self, **kwargs):
        """
        Обновление конфигурации
        
        Args:
            **kwargs: параметры
        """
        self.config.update(**kwargs)
        self.config.save(self.config_path)
        
    def reset_config(self):
        """Сброс конфигурации"""
        self.config = Config()
        self.config.save(self.config_path)
        
    def get_env_config(self) -> Dict[str, Any]:
        """
        Получение конфигурации из переменных окружения
        
        Returns:
            Dict[str, Any]: конфигурация
        """
        env_config = {}
        
        # NLP
        if os.environ.get("NLP_MODEL_NAME"):
            env_config["nlp"] = {
                "model_name": os.environ["NLP_MODEL_NAME"]
            }
        if os.environ.get("USE_CUDA"):
            env_config.setdefault("nlp", {})
            env_config["nlp"]["device"] = (
                "cuda" if os.environ["USE_CUDA"] == "1" else "cpu"
            )
            
        # Storage
        if os.environ.get("DB_PATH"):
            env_config["storage"] = {
                "db_path": os.environ["DB_PATH"]
            }
        if os.environ.get("CACHE_DIR"):
            env_config.setdefault("storage", {})
            env_config["storage"]["cache_dir"] = os.environ["CACHE_DIR"]
            
        # Logging
        if os.environ.get("LOG_DIR"):
            env_config["logging"] = {
                "log_dir": os.environ["LOG_DIR"]
            }
        if os.environ.get("LOG_LEVEL"):
            env_config.setdefault("logging", {})
            env_config["logging"]["log_level"] = os.environ["LOG_LEVEL"]
            
        # API
        if os.environ.get("API_RATE_LIMIT_CALLS"):
            env_config["api"] = {
                "rate_limit_calls": int(os.environ["API_RATE_LIMIT_CALLS"])
            }
        if os.environ.get("API_RATE_LIMIT_PERIOD"):
            env_config.setdefault("api", {})
            env_config["api"]["rate_limit_period"] = int(
                os.environ["API_RATE_LIMIT_PERIOD"]
            )
            
        # Agent
        if os.environ.get("MAX_TASKS"):
            env_config["agent"] = {
                "max_tasks": int(os.environ["MAX_TASKS"])
            }
        if os.environ.get("TASK_TIMEOUT"):
            env_config.setdefault("agent", {})
            env_config["agent"]["task_timeout"] = int(
                os.environ["TASK_TIMEOUT"]
            )
            
        # Debug
        if os.environ.get("DEBUG"):
            env_config["debug"] = os.environ["DEBUG"] == "1"
            
        return env_config
        
    def load_env_config(self):
        """Загрузка конфигурации из переменных окружения"""
        env_config = self.get_env_config()
        if env_config:
            self.update_config(**env_config) 