from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel, tool
from typing import Dict, Any, List, Optional
import os
from datetime import datetime

model_search = OpenAIServerModel(
    model_id="google/gemini-2.0-flash-001",
    api_base="https://api.vsegpt.ru/v1", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("OPENAI_API_KEY"), # Switch to the API key for the server you're targeting.
    max_tokens=150000,
)

model_lite = OpenAIServerModel(
    model_id="openai/gpt-4o-mini",
    api_base="https://api.vsegpt.ru/v1", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("OPENAI_API_KEY"), # Switch to the API key for the server you're targeting.
    max_tokens=80000,
)

model_code = OpenAIServerModel(
    model_id="openai/o3-mini",
    api_base="https://api.vsegpt.ru/v1", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("OPENAI_API_KEY"), # Switch to the API key for the server you're targeting.
    max_tokens=60000,
)

model_hard = OpenAIServerModel(
    model_id="openai/o3-mini",
    #model_id="openai/gpt-4o-mini",
    api_base="https://api.vsegpt.ru/v1", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("OPENAI_API_KEY"), # Switch to the API key for the server you're targeting.
    max_tokens=60000,
)

class AgentFactory:
    """Фабрика для динамического создания агентов с разными профилями"""
    
    AGENT_PROFILES = {
        'researcher': {
            'model': model_search,
            'tools': [DuckDuckGoSearchTool()],
            'description': f'Собрать актуальную информацию по заданной теме, используя доступные источники. Предоставить полные и точные структурированные данные. Ты не выполняешь никаких действий, ты только собираешь информацию.\nТекущие дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'dependencies': []  # нет зависимостей
        },
        'analyst': {
            'model': model_hard,
            'tools': ['data_analysis'],
            'description': f'На основе собранных данных выявить ключевые тренды и паттерны. Подготовить подробный аналитический отчет. Ты должен быть очень внимательным к деталям и точности данных. Ты даешь только точные данные и не принимаешь никаких предположений, а так же не выполняешь никаких действий, ты только анализируешь информацию.\nТекущие дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'dependencies': ['researcher']  # зависит от researcher
        },
        'visualizer': {
            'model': model_code,
            'tools': [],
            'description': f'''Создать наглядные визуализации для выявленных трендов и закономерностей. 
            Текущие дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            ВАЖНО: 
            1. Все графики должны создаваться через инструмент data_visualization
            2. Графики сохраняются в файл в каталоге plots
            3. Не использовать plt.show()
            4. Использовать только plt.savefig()
            5. Данные должны быть в формате словаря
            6. Используйте seaborn для стилизации
            7. Добавить легенду и подписи на русском языке
            ''',
            'dependencies': ['analyst']  # зависит от analyst
        },
        'validator': {
            'model': model_lite,
            'tools': ['fact_checking'],
            'description': f'Проверить достоверность источников и валидировать сделанные выводы.\nТекущие дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'dependencies': ['researcher', 'analyst']  # зависит от researcher и analyst
        },
        'architect': {
            'model': model_hard,
            'tools': [],
            'description': f"""Проектировать архитектуру системы, создавать описание компонентов и их взаимодействия.
            Текущие дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Инструкция:
            - Подготовить план и описание архитектуры системы
            - Детально описать каждый компонент системы
            - Определить их взаимодействие и взаимосвязи
            """,
            'dependencies': ['researcher', 'analyst']  # зависит от researcher и analyst для анализа требований
        },
        'diagram_creator': {
            'model': model_code,
            'tools': [],
            'description': '''Создать диаграмму в формате PlantUML
            - Тип диаграммы необходимо определить в зависимости от задачи
            - Используй визуальные стили для диаграммы для лучшей читаемости
            - Добавь легенду и подписи
            - Добавь заметки и комментарии
            - Весь текст должен быть на русском языке
            - Ни в коем случае нельзя переносить текстовые строки в ковычках на другую строку, это сломает диаграмму, если необходимо, можно сократить текст, но нельзя переносить!
            - Вывести сгенерированный код в формате PlantUML без дополнительных символов и комментариев

            Визуальное оформление текста (Creole):
            - **жирный текст** - окружить двумя звездочками
            - //курсив// - окружить двумя слешами
            - ""моноширинный шрифт"" - окружить двумя кавычками
            - --зачеркнутый-- - окружить двумя дефисами
            - __подчеркнутый__ - окружить двумя подчеркиваниями
            - ~~волнистое подчеркивание~~ - окружить двумя тильдами
            
            Списки:
            * Маркированный список
            * Второй пункт
            ** Подпункт
            # Нумерованный список
            # Второй пункт
            ## Подпункт
            
            Горизонтальные линии:
            ---- (обычная линия)
            ==== (двойная линия)
            ____ (жирная линия)
            .... (пунктирная линия)
            
            Заголовки:
            = Очень большой заголовок
            == Большой заголовок
            === Средний заголовок
            ==== Маленький заголовок

            Работа с цветами:
            - Цвет текста: <color:red>красный текст</color>
            - Цвет фона: <back:yellow>текст с желтым фоном</back>
            - Цвет рамки: <#red>красная рамка</color>
            
            Доступные цвета:
            - Базовые: red, blue, green, yellow, orange, black, white
            - HEX: <color:#FF0000>красный</color>, <color:#00FF00>зеленый</color>
            - RGB: <color:rgb(255,0,0)>красный</color>
            Тема оформления:
            !theme materia
            
            Пример цветовой схемы для элементов:
            skinparam {
              BackgroundColor white
              BorderColor black
              FontColor blue
              FontSize 12
              FontName Arial
            }
            
            Цвета для отдельных элементов:
            class Example {
              + method()
            }
            Example : {
              BackgroundColor PaleGreen
              BorderColor DarkGreen
              FontColor DarkBlue
            }

            Иконки и символы:
            - Эмодзи: <:1f600:> (смайлик), <:sunny:> (солнце), можно использовать любые эмодзи
            - Изменение цвета эмодзи: <#green:sunny:> (зеленое солнце)
            
            Стандартные иконки:
            - <&check> - галочка
            - <&cross> - крестик
            - <&star> - звезда
            - <&cloud> - облако
            - <&folder> - папка
            - <&file> - файл
            - <&cog> - шестеренка
            - <&person> - человек
            - <&home> - дом
            - <&clock> - часы
            - <&search> - поиск
            - <&settings> - настройки
            - <&warning> - предупреждение
            - <&info> - информация
            - <&key> - ключ
            - <&lock> - замок
            - <&unlock> - открытый замок
            - <&link> - ссылка
            - <&heart> - сердце
            - <&trash> - корзина
            
            Использование иконок в элементах:
            class Example << (F,#FF7700) >>
            note right of Example : Можно использовать <&key> внутри заметок
            
            Размещение иконок:
            - В заголовках: == <&folder> Заголовок
            - В заметках: note left : <&info> Важная информация
            - В классах: class Example << (C,#FF7700) >> {
              + <&key> secureMethod()
            }
            - Пример в легенде: legend right
              <&check> Выполнено
              <&cross> Не выполнено
            end legend

            Использование элементов:
              legend
                Some legend
              end legend
              header: some header
              footer: some footer
              caption: some caption
              note left of <element>, note right of <element>, note top of <element>, note bottom of <element> : Some note\\n for <element>

            Типы диаграмм:
            'gantt_chart': 
              - @startgantt/@endgantt
              - Project starts
              - Задачи в формате [Task] as [T1] lasts X days
              - Связи между задачами
            'mind_map': 
              - @startmindmap/@endmindmap
              - Центральный узел с *
              - Ветви с разным количеством *
              - Цветовое оформление
              - Заголовок: {title}
            'flowchart':
              Используй следующие элементы:
              - @startuml/@enduml
              - start/stop
              - if/then/else при необходимости
              - Действия в формате :action;
              - Стрелки --> для связей
              - Описание: {description}
              - Заголовок: {title}
            'project_timeline':
              - Используй следующие элементы:
              - @startuml/@enduml
              - Временные метки @0, @5 и т.д.
              - Описания событий
              - Стилизация и цвета
              - Описание: {description}
              - Заголовок: {title}
            'infographic':
              - Используй следующие элементы:
              - @startuml/@enduml
              - rectangle для блоков
              - Вложенные элементы
              - Цветовое оформление
              - Разные формы (rectangle, circle, artifact)
              - Описание: {description}
              - Заголовок: {title}
            'org_chart':
              - Используй следующие элементы:
              - @startuml/@enduml
              - Связи между элементами -->
              - Стилизация узлов
              - Группировка при необходимости
              - Описание: {description}
              - Заголовок: {title}
            'process_diagram':
              - Используй следующие элементы:
              - @startuml/@enduml
              - start/stop
              - Действия в формате :action;
              - Условия при необходимости
              - Параллельные процессы при необходимости
              - Описание: {description}
              - Заголовок: {title}  
            'component_diagram':
              - Используй следующие элементы:
              - @startuml/@enduml
              - Компоненты в формате [Component] или component Component
              - Интерфейсы в формате () "Interface" или interface Interface
              - Связи между компонентами:
                * --> для направленных связей
                * -- для ненаправленных связей
                * ..> для пунктирных направленных связей
                * .. для пунктирных ненаправленных связей
              - Группировка компонентов:
                * package "Имя пакета" { ... }
                * node "Имя узла" { ... }
                * folder "Имя папки" { ... }
                * frame "Имя фрейма" { ... }
                * cloud { ... }
                * database { ... }
              - Порты:
                * port для обычных портов
                * portin для входных портов
                * portout для выходных портов
              - Заметки:
                * note left of, note right of, note top of, note bottom of
              - Описание: {description}
              - Заголовок: {title}
            'entity_relationship':
              - Используй следующие элементы:
              - @startuml/@enduml
              - Сущности определяются через entity "Имя" as E01 { ... }
              - Обязательные атрибуты помечаются *
              - Типы связей:
                * |o-- : Ноль или один
                * ||-- : Ровно один
                * }o-- : Ноль или много
                * }|-- : Один или много
              - Дополнительные параметры:
                * hide circle - скрыть точки
                * skinparam linetype ortho - для правильного отображения связей
              - Атрибуты сущностей:
                * * attribute_name : type - обязательный атрибут
                * attribute_name : type - необязательный атрибут
                * -- для разделения секций
                * <<generated>> для автогенерируемых полей
                * <<FK>> для внешних ключей
              - Примеры атрибутов:
                * *id : number <<generated>>
                * *name : text
                * description : text
                * *foreign_key : number <<FK>>
              - Описание: {description}
              - Заголовок: {title}
            ''' + f'\nТекущие дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'dependencies': ['architect']  # зависит от architect
        }
    }

    def __init__(self):
        self.agent_counter = 1
        self.active_agents = {}
        self.manager_agent = None
        self.dependencies_graph = {}  # Граф зависимостей между агентами

    def _create_tool(self, tool_name: str):
        """Динамическое создание инструментов по имени"""
        print(f"Создание инструмента: {tool_name}")  # Добавляем логирование
        if isinstance(tool_name, str):
            if tool_name == 'data_analysis':
                @tool
                def data_analysis(data: Dict, query: str) -> str:
                    """Анализирует данные и возвращает результаты анализа.
                    
                    Args:
                        data: Словарь с данными для анализа в формате ключ-значение
                        query: Строка запроса, определяющая тип анализа
                    
                    Returns:
                        str: Результаты анализа в текстовом формате
                    """
                    from pandas import DataFrame
                    df = DataFrame(data)
                    return f"Анализ: {df.describe()}\nТренды: {df.mean().to_dict()}"
                return data_analysis
                        
            elif tool_name == 'fact_checking':
                @tool
                def fact_checking(claim: str, sources: List[str]) -> str:
                    """Проверяет достоверность утверждения по указанным источникам.
                    
                    Args:
                        claim: Утверждение для проверки
                        sources: Список источников для проверки утверждения
                    
                    Returns:
                        str: Результат проверки достоверности
                    """
                    return f"Проверка '{claim}' по {len(sources)} источникам"
                return fact_checking
                        
            #raise ValueError(f"Неизвестный тип инструмента: {tool_name}")
        
        # Если tool_name это уже инструмент (например DuckDuckGoSearchTool)
        return tool_name

    def create_manager_agent(self, model) -> CodeAgent:
        """Создание менеджер-агента, который может управлять другими агентами
        
        Args:
            model: Модель для агента
            
        Returns:
            CodeAgent: Настроенный менеджер-агент
        """
        manager = CodeAgent(
            tools=[],  # У менеджера нет собственных инструментов
            model=model,
            max_steps=15,
            name="manager",
            description="Менеджер-агент, координирующий работу других агентов",
            managed_agents=[],  # Список будет заполнен позже
            additional_authorized_imports=["time", "numpy", "pandas"]
        )
        self.manager_agent = manager
        return manager

    def _validate_dependencies(self, profile_type: str) -> bool:
        """Проверяет, что все зависимости агента уже созданы
        
        Args:
            profile_type: Тип профиля агента
            
        Returns:
            bool: True если все зависимости удовлетворены, False иначе
        """
        if profile_type not in self.AGENT_PROFILES:
            return False
            
        dependencies = self.AGENT_PROFILES[profile_type].get('dependencies', [])
        for dep in dependencies:
            if not any(agent.name == dep for agent in self.active_agents.values()):
                return False
        return True

    def _update_dependencies_graph(self, profile_type: str):
        """Обновляет граф зависимостей при создании нового агента
        
        Args:
            profile_type: Тип профиля созданного агента
        """
        dependencies = self.AGENT_PROFILES[profile_type].get('dependencies', [])
        self.dependencies_graph[profile_type] = dependencies

    def create_agent(self, profile_type: str) -> CodeAgent:
        """Создание агента заданного типа
        
        Args:
            profile_type: Тип профиля агента из AGENT_PROFILES
            
        Returns:
            CodeAgent: Настроенный агент с указанным профилем
            
        Raises:
            ValueError: Если не удовлетворены зависимости агента
        """
        if profile_type not in self.AGENT_PROFILES:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        profile = self.AGENT_PROFILES[profile_type]
        tools = [self._create_tool(tool) for tool in profile['tools']]
        
        if any(tool is None for tool in tools):
            print(f"ВНИМАНИЕ: Не все инструменты были созданы для профиля {profile_type}")
            print(f"Инструменты: {tools}")
        
        # Создаем агента с name и description для возможности управления им
        agent = CodeAgent(
            tools=tools,
            model=profile['model'],
            max_steps=15,
            name=profile_type,
            description=profile['description'],
            additional_authorized_imports=["time", "numpy", "pandas", "requests", "meteostat", "matplotlib", 
                                        "networkx", "seaborn", "scipy", "scipy.stats", "scipy.stats.skew", 
                                        "scipy.stats.kurtosis", "scipy.stats.probplot"],
        )
        
        # Добавляем agent_id и обновляем граф зависимостей
        agent_id = f"{profile_type}-{self.agent_counter}"
        setattr(agent, 'agent_id', agent_id)
        self._update_dependencies_graph(profile_type)
        
        self.active_agents[agent_id] = agent
        self.agent_counter += 1

        # Если есть менеджер-агент, добавляем нового агента в его управляемые агенты
        if self.manager_agent is not None:
            if not hasattr(self.manager_agent, 'managed_agents'):
                self.manager_agent.managed_agents = []
            self.manager_agent.managed_agents.append(agent)
        
        return agent

    def get_agent(self, name: str) -> Optional[CodeAgent]:
        """Получить агента по имени
        
        Args:
            name: Имя агента
            
        Returns:
            Optional[CodeAgent]: Найденный агент или None
        """
        for agent in self.active_agents.values():
            if agent.name == name:
                return agent
        return None

    def get_agent_dependencies(self, profile_type: str) -> List[str]:
        """Получить список зависимостей агента
        
        Args:
            profile_type: Тип профиля агента
            
        Returns:
            List[str]: Список имен агентов, от которых зависит данный агент
        """
        if profile_type not in self.AGENT_PROFILES:
            return []
        return self.AGENT_PROFILES[profile_type].get('dependencies', [])
