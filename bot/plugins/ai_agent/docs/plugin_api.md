# API системы плагинов

## Обзор

Система плагинов позволяет расширять функциональность AI Agent с помощью внешних модулей. Каждый плагин представляет собой отдельный Python-пакет, который может добавлять новые возможности или модифицировать существующие.

## Структура плагина

Каждый плагин должен иметь следующую структуру:

```
plugin_name/
├── metadata.json
├── plugin.py
├── requirements.txt
└── README.md
```

### metadata.json

Файл метаданных плагина в формате JSON:

```json
{
    "name": "plugin_name",
    "version": "1.0.0",
    "description": "Plugin description",
    "author": "Author Name",
    "dependencies": {
        "package_name": ">=1.0.0,<2.0.0"
    },
    "entry_point": "plugin",
    "config_schema": {
        "type": "object",
        "properties": {
            "setting1": {"type": "string"},
            "setting2": {"type": "number"}
        }
    },
    "enabled": true
}
```

### plugin.py

Основной файл плагина:

```python
from bot.plugins.ai_agent.plugin_system import Plugin, PluginMetadata

class Plugin(Plugin):
    async def initialize(self):
        # Инициализация плагина
        pass
        
    async def shutdown(self):
        # Завершение работы плагина
        pass
        
    def configure(self, config):
        # Конфигурация плагина
        self.config = config
```

## API плагина

### Методы базового класса

- `initialize()` - инициализация плагина
- `shutdown()` - завершение работы плагина
- `configure(config)` - конфигурация плагина

### Доступные свойства

- `metadata` - метаданные плагина
- `config` - конфигурация плагина

## Управление плагинами

### Загрузка плагина

```python
plugin_manager = PluginManager()
await plugin_manager.load_plugin("path/to/plugin")
```

### Выгрузка плагина

```python
await plugin_manager.unload_plugin("plugin_name")
```

### Включение/отключение плагина

```python
await plugin_manager.enable_plugin("plugin_name")
await plugin_manager.disable_plugin("plugin_name")
```

### Конфигурация плагина

```python
plugin_manager.configure_plugin("plugin_name", {
    "setting1": "value1",
    "setting2": 42
})
```

## Версионирование

Система поддерживает семантическое версионирование (SemVer). При загрузке новой версии плагина происходит проверка версии и обновление только если новая версия больше текущей.

## Зависимости

Зависимости указываются в формате, совместимом с pip:

```
package_name>=1.0.0,<2.0.0
```

При загрузке плагина происходит проверка всех зависимостей.

## Безопасность

- Плагины выполняются в том же процессе, что и основное приложение
- Необходимо тщательно проверять плагины перед установкой
- Рекомендуется использовать только доверенные источники

## Примеры

### Минимальный плагин

```python
from bot.plugins.ai_agent.plugin_system import Plugin, PluginMetadata

class Plugin(Plugin):
    async def initialize(self):
        self.logger.info("Plugin initialized")
```

### Плагин с конфигурацией

```python
class Plugin(Plugin):
    def configure(self, config):
        self.api_key = config.get("api_key")
        self.max_retries = config.get("max_retries", 3)
```

### Плагин с зависимостями

metadata.json:
```json
{
    "name": "nlp_plugin",
    "version": "1.0.0",
    "dependencies": {
        "nltk": ">=3.6.0",
        "spacy": ">=3.0.0"
    }
}
``` 