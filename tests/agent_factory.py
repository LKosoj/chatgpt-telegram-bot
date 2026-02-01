from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel, tool
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
import re
from requests.exceptions import RequestException

model_search = OpenAIServerModel(
    model_id="google/gemini-2.0-flash-001",
    api_base="https://api.vsegpt.ru/v1", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("OPENAI_API_KEY"), # Switch to the API key for the server you're targeting.
    max_tokens=650000,
    extra_headers={ "X-Title": "CAgent" },
)

model_lite = OpenAIServerModel(
    model_id="openai/gpt-4o-mini",
    api_base="https://api.vsegpt.ru/v1", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("OPENAI_API_KEY"), # Switch to the API key for the server you're targeting.
    max_tokens=20000,
    extra_headers={ "X-Title": "CAgent" },
)

model_code = OpenAIServerModel(
    model_id="openai/o3-mini",
    api_base="https://api.vsegpt.ru/v1", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("OPENAI_API_KEY"), # Switch to the API key for the server you're targeting.
    max_tokens=20000,
    extra_headers={ "X-Title": "CAgent" },
)

model_hard = OpenAIServerModel(
    #model_id="openai/o3-mini",
    model_id="openai/o3-mini-high",
    api_base="https://api.vsegpt.ru/v1", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("OPENAI_API_KEY"), # Switch to the API key for the server you're targeting.
    max_tokens=20000,
    extra_headers={ "X-Title": "CAgent" },
)

AUTHORIZED_IMPORTS = [
  "requests",
  "zipfile",
  "os",
  "pandas",
  "numpy",
  "sympy",
  "json",
  "bs4",
  "pubchempy",
  "xml",
  "yahoo_finance",
  "Bio",
  "sklearn",
  "pydub",
  "scikit-learn",
  "io",
  "PIL",
  "chess",
  "PyPDF2",
  "pptx",
  "torch",
  "datetime",
  "fractions",
  "csv",
  "time",
  "meteostat",
  "matplotlib",
  "networkx",
  "seaborn",
  "scipy",
  "yfinance",
  "coinmarketcap",
  "coinpaprika",
  "coinbase",
  "pandas_market_calendars",
]

AGENT_PROFILES = {
    'researcher': {
        'model': model_search,
        'tools': [DuckDuckGoSearchTool(), 'webpage_content'],
        'type': 'tool_calling',
        'description': f'Собиратель актуальной информации из интернета. Использует поисковые системы и инструменты для извлечения данных, обеспечивая точность и структурированность информации. Используйте его для получения информации из интернета.',
        'prompt_templates': f'''
         Инструкция:
         - Соберите актуальную и подробную информацию по заданной теме, используя доступные интернет-источники.
         - Все данные должны быть точными и структурированными.
         - Если информации недостаточно, обязательно укажите это в ответе. Без этого ответ будет считаться неполным и не будет принят.
         - Вы выполняете только поиск информации, не выполняете никаких действий, не принимаете никаких решений, не выполняете никаких действий, ты только ищешь информацию.
         - Если нужно использовать библиотеку sklearn, используй библиотеку scikit-learn вместо sklearn
        ''',
        'dependencies': [],
    },
    'analyst': {
        'model': model_hard,
        'tools': [],
        'description': f'Аналитический агент, анализирующий данные и формирующий подробный отчет с ключевыми выводами и тенденциями. Используйте его для анализа данных, текста и формирования отчетов.',
        'prompt_templates': f'''
         Инструкция:
         - Проведите глубокий анализ переданных данных и сформируйте детальный отчет с выводами и рекомендациями.
         - Используйте только достоверную информацию.
         - В случае недостатка данных, укажите это в отчете.
         - Если нужно использовать библиотеку sklearn, используй библиотеку scikit-learn вместо sklearn
         - Если при анализе будет ошибка - прежде чем продолжить выполнение кода - устрани ее!
         - Если в запросе пользователя необходима визуализация данных, не строй графики! Это сделает другой агент!
        ''',
        'dependencies': ['researcher'],
    },
    'visualizer': {
        'model': model_code,
        'tools': [],
        'description': f'Агент визуализации данных, создающий наглядные графики для анализа результатов. Используйте его для визуализации данных.',
        'prompt_templates': f'''
         Инструкция:
         - Сгенерируйте визуализации (графики, диаграммы) на основе предоставленных данных.
         - Используйте библиотеку matplotlib для создания графиков и диаграмм.
         - Используйте библиотеку seaborn для стилизации графиков и диаграмм.
         - Сохраните графики в файлы в каталоге 'plots', используя метод plt.savefig(). Имя файла должно вклюать значение'session_id', например: 'plots/plot_a534b3c9.png'
         - Не используйте plt.show(); все подписи, легенда и заголовки должны быть на русском языке.
         - Если нужно использовать библиотеку sklearn, используй библиотеку scikit-learn вместо sklearn
        ''',
        'dependencies': ['analyst'],
    },
    'validator': {
        'model': model_search,
        'tools': [DuckDuckGoSearchTool(), 'webpage_content'],
        'description': f'Агент проверки достоверности, валидирующий источники данных и результаты анализа. Используйте его для проверки достоверности данных.',
        'prompt_templates': f'''
         Инструкция:
         - Проверьте достоверность предоставленных источников и выводов.
         - При недостатке данных или обнаружении несоответствий, сообщите об этом.
        ''',
        'dependencies': ['researcher'],
    },
    'architect': {
        'model': model_hard,
        'tools': [],
        'description': f'Агент архитектурного проектирования, разрабатывающий детальное описание системы, её компонентов и их взаимодействия. Используйте его для проектирования системы, не для диаграмм.',
        'prompt_templates': f'''
         Инструкция:
         - Разработайте подробное описание архитектуры системы, включая описание компонентов, их функции и взаимосвязи.
         - Обеспечьте структурированное и понятное представление технической архитектуры.
        ''',
        'dependencies': ['analyst'],
    },
    'diagram_creator': {
        'model': model_code,
        'tools': [],
        'description': '''Агент генерации диаграмм в формате PlantUML. Создает диаграммы по заданному техническому заданию, соблюдая правила стилизации и форматирования. Используйте его для генерации диаграмм.''',
        'prompt_templates': '''
         Инструкция:
         - На вход подается техническое задание, на основе которого необходимо создать диаграмму в формате PlantUML
         - Тип диаграммы необходимо определить в зависимости от задачи. Внимательно изучи техническое задание и определи какой тип диаграммы нужен, если в задании указан тип диаграммы, то используй его, если не указан, то определи какой тип диаграммы нужен на основе задачи.
         - Используй визуальные стили для диаграммы для лучшей читаемости
         - Используй emoji в формате <#0:sunglasses:>
         - Добавь легенду и подписи
         - Добавь заметки и комментарии
        - Весь текст должен быть на русском языке
        - **Ни в коем случае нельзя переносить текстовые строки, заключенные в кавычки, на другую строку, это сломает диаграмму. Если текст длинный, его необходимо сократить, сделав его не длиннее 50 символов и не потеряв смысл, но нельзя переносить, это обязательное условие, очень важно!**
        - Вывести сгенерированный код в формате PlantUML

# PlantUML Guid

## 1. Basic Syntax and Structure

*   **Diagram Start and End:** All PlantUML diagrams must begin with `@startuml` and end with `@enduml`.   

    
  @startuml    
  ' Your diagram code here 
  @enduml
      

*   **Comments:** Use single quotes (`'`) for comments.    

    
  @startuml    
  ' This is a comment
  @enduml
      

## 2. Diagram Types    

PlantUML supports the following diagram types:     

*   **Sequence Diagram:** Illustrates interactions between objects in a sequence.  
*   **Class Diagram:** Shows the structure of classes and their relationships.     
*   **Use Case Diagram:** Depicts the interactions between actors and the system.  
*   **Activity Diagram:** Models the flow of activities in a process.  
*   **Component Diagram:** Represents the components of a system and their dependencies. 
*   **Deployment Diagram:** Shows the physical deployment of software components.  
*   **State Diagram:** Models the states of an object and the transitions between them.  
*   **Object Diagram:** Shows instances of classes and their relationships at a specific point in time.    
*   **Timing Diagram:** Illustrates the timing constraints of interactions.  
*   **Network Diagram:** Visualizes the structure of a computer network.     
*   **Wireframe Diagram:** Creates basic UI sketches.
*   **Archimate Diagram:** Models enterprise architectures using the ArchiMate language. 
*   **SDL Diagram:** Specifies the behavior of reactive and distributed systems.   
*   **Block Diagram:** Shows the organization of elements in blocks.   
*   **Ditaa Diagram:** Integrates Ditaa diagrams.    
*   **Salt Diagram:** Creates user interface mocks using Salt syntax. 
*   **Mind Map:** Creates mind maps.
*   **Gantt Chart:** Creates gantt charts.

Let's look into some specific Diagram syntaxes.

### 2.1 Sequence Diagrams    


*   **Participants:** Participants are automatically declared. You can also explicitly define them using the     
  `participant` or `actor` keyword. The only difference is in the rendering style.   

    
  @startuml    
  participant Alice  
  actor Bob    
  Alice -> Bob: Authentication Request 
  Bob --> Alice: Authentication Response     
  @enduml
      

*   **Actors:** Actors are similar to participants but are typically used to represent users or external systems.

    
  @startuml    
  actor User   
  User -> System: Makes a request
  System --> User: Returns a response  
  @enduml
      

*   **Messages:** Use `->`, `-->`, `<-`, `<--` to represent solid and dotted arrows.  You can also use `-o` (arrow     
  with circle at the end), `-x` (arrow with x at the end).   

    
  @startuml    
  Alice -> Bob: Message    
  Alice --> Bob: Dotted Message  
  Bob <- Alice: Reply
  Bob <-- Alice: Dotted Reply    
  Alice -o Bob : Circle at the End     
  Alice -x Bob : X at the end    
  @enduml
      

*   **Aliases:** Use the `as` keyword to define aliases for participants.

    
  @startuml    
  participant "Very Long Name" as VLN  
  VLN -> VLN: Self-Message 
  @enduml
      

*   **Activation/Deactivation:**  Use `activate` and `deactivate` to show processing time. 

    
  @startuml    
  Alice -> Bob: Request    
  activate Bob 
  Bob --> Alice: Response  
  deactivate Bob     
  @enduml
      

*   **Notes:** Add notes using the `note left`, `note right`, or `note over` keywords.

  
  @startuml
  Alice -> Bob: Request
  note right: This is a note
  @enduml
  

*   **Lifelines:** Lifelines are automatically drawn. You can destroy them using `destroy` .   

    
  @startuml    
  Alice -> Bob: Request    
  activate Bob 
  Bob --> Alice: Response  
  deactivate Bob     
  destroy Bob  
  @enduml
      

*   **Ref:** Reference a section in another diagram  

    
  @startuml    
  participant Alice  
  actor Bob    
  ref over Alice, Bob : init     
  Alice -> Bob : hello     
  ref over Bob 
  This can be on several lines   
  end ref
  @enduml
      

### 2.2 Class Diagrams 

*   **Classes:** Define classes using the `class` keyword.

  
  @startuml
  class MyClass {    
- attribute1: type   
+ method1()    
  }
  @enduml
      

*   **Abstract Classes:** Use the `abstract` keyword.

  
  @startuml
  abstract class AbstractClass { 
    abstract method()     
  }
  @enduml
      

*   **Interfaces:** Use the `interface` keyword.     

    
  @startuml    
  interface MyInterface {  
    method() 
  }
  @enduml
  

*   **Relationships:** Define relationships between classes using:     

  *   `--`: Association    
  *   `*--`: Composition   
  *   `o--`: Aggregation   
  *   `<|--`: Inheritance  
  *   `--o`: Part of 

    
  @startuml    
  class ClassA 
  class ClassB 
  ClassA -- ClassB: Relationship 
  @enduml
      

### 2.3 Use Case Diagrams    

*   **Actors:** Define actors using the `actor` keyword.

  
  @startuml    
  actor User   
  @enduml
      

*   **Use Cases:** Define use cases using parentheses.     

    
  @startuml    
  (Use Case)   
  @enduml
      

*   **Relationships:** Use `--`, `<<includes>>`, `<<extends>>` to define relationships.  

  
  @startuml    
  actor User   
  (Use Case) -- User 
  (Use Case) <<includes>> (Another Use Case) 
  @enduml
      

### 2.4 Activity Diagrams

*   **Start and End:** Use `(*)` to represent the start and end states.

    
  @startuml    
  (*) --> Activity   
  Activity --> (*)   
  @enduml
      

*   **Activities:** Define activities using rectangles.    

    
  @startuml    
  start  
  :Activity;   
  stop   
  @enduml
      

*   **Decisions:** Use diamonds for decision points.

    
  @startuml    
  start  
  if (Condition) then (yes)
    :Activity 1;     
  else (no)    
    :Activity 2;     
  endif  
  stop   
  @enduml
      

*   **Forks and Joins:** Use `fork` and `join` to represent parallel activities.   


    
  @startuml    
  start  
  fork   
    :Activity 1;     
  fork again   
    :Activity 2;     
  end fork     
  stop   
  @enduml
      

### 2.5 Mind Map

    
    @startmindmap
    * count
    ** 100
    *** 101
    *** 102
    ** 200

    left side

    ** A
    *** AA
    *** AB
    ** B
    @endmindmap
  
Команды чтобы изменить направление развёртки всей диаграммы:
left to right direction (by default)
top to bottom direction
right to left direction

Example with Creole syntax:
    
    @startmindmap
    * Creole on Mindmap
    left side
    **:==Creole
      This is **bold**
      This is //italics//
      This is ""monospaced""
      This is --stricken-out--
      This is __underlined__
      This is ~~wave-underlined~~
    --test Unicode and icons--
      This is <U+221E> long
      This is a <&code> icon
      Use image : <img:http://plantuml.com/logo3.png>
    ;
    **: <b>HTML Creole 
      This is <b>bold</b>
      This is <i>italics</i>
      This is <font:monospaced>monospaced</font>
      This is <s>stroked</s>
      This is <u>underlined</u>
      This is <w>waved</w>
      This is <s:green>stroked</s>
      This is <u:red>underlined</u>
      This is <w:#0000FF>waved</w>
    -- other examples --
      This is <color:blue>Blue</color>
      This is <back:orange>Orange background</back>
      This is <size:20>big</size>
    ;
    right side
    **:==Creole line
    You can have horizontal line
    ----
    Or double line
    ====
    Or strong line
    ____
    Or dotted line
    ..My title..
    Or dotted title
    //and title... //
    ==Title==
    Or double-line title
    --Another title--
    Or single-line title
    Enjoy!;
    **:==Creole list item
    **test list 1**
    * Bullet list
    * Second item
    ** Sub item
    *** Sub sub item
    * Third item
    ----
    **test list 2**
    # Numbered list
    # Second item
    ## Sub item
    ## Another sub item
    # Third item
    ;
    @endmindmap

### 2.6 Gantt Chart

    
    @startgantt
    Project starts
    [Task] as [T1] lasts X days
    [Task] as [T2] lasts X days
    [Task] as [T3] lasts X days
    @endgantt
  

    
  @startgantt
  [Стартовое совещание] requires 1 days and is colored in blue
  then [Разработка прототипа] requires 5 days
  [Тестирование] requires 4 days
  [Тестирование] starts at [Разработка прототипа]'s end
  [Разработка прототипа] is colored in Green
  [Тестирование] is colored in gray

  legend
  Легенда:
  |= Цвет |= Тип задачи |
  |<#gray> | Запланированные |
  |<#Green>| В работе |
  |<#blue> | Завершённые |
  end legend

  @endgantt
  

    
  @startgantt
  [Prototype design] requires 13 days
  [Test prototype] requires 4 days
  [Test prototype] starts at [Prototype design]'s end
  [Prototype design] is colored in Fuchsia/FireBrick
  [Test prototype] is colored in GreenYellow/Green
  @endgantt


### 3. Component Diagrams

PlantUML provides several ways to customize the appearance of diagrams.

*   **Skinparams:** Use `skinparam` to set global styling options.

    
  @startuml    
  skinparam backgroundColor #EEEBDC    
  skinparam defaultFontName Arial
  skinparam sequenceArrowThickness 2   
  Alice -> Bob: Message    
  @enduml
      

*   **Colors:** Use predefined color names or hexadecimal color codes. 

    
  @startuml    
  participant Alice #red   
  participant Bob #99FF99  
  Alice -> Bob: Message    
  @enduml
      

*   **Fonts:** Change the default font using `skinparam defaultFontName <font_name>`.    

*   **Borders and Fills:** Customize borders and fills using skinparams.     

    
  @startuml    
  skinparam classBorderColor blue
  skinparam classBackgroundColor lightblue   
  class MyClass
  @enduml
  

*   **Shadows:** Add shadows using `skinparam Shadowing true`.   

*   **Gradients:** Define gradients using color combinations and direction indicators.   

    
  @startuml    
  skinparam participantBackgroundColor #CCCCFF/#DDDDFF   
  participant Alice  
  @enduml
      

## 4. Icons and Emojis 

*   **Built-in Icons (OpenIconic):** Use `<$icon_name>` syntax.  

    
  @startuml    
  actor <$user> User 
  User -> System: Login    
  @enduml
      

*   **MaterialDesign Icons:** Available through specific libraries.    

*   **Emojis:** Use Unicode emojis directly in the diagram.  Support varies depending on the PlantUML renderer.  

## 5. Important Notes  

*   PlantUML is case-insensitive for keywords but case-sensitive for identifiers (e.g., class names).
*   Whitespace is generally ignored.
        ''',
        'dependencies': [['architect', 'analyst'],]
    }
}

class AgentFactory:
    """Фабрика для динамического создания агентов с разными профилями"""
    

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
                        
            elif tool_name == 'webpage_content':
                @tool
                def webpage_content(url: str) -> str:
                    """Visits a webpage at the given URL and returns its content as a markdown string.

                  Args:
                      url: The URL of the webpage to visit.

                  Returns:
                      The content of the webpage converted to Markdown, or an error message if the request fails.
                  """
                    try:
                        # Send a GET request to the URL
                        response = requests.get(url)
                        response.raise_for_status()  # Raise an exception for bad status codes

                        # Convert the HTML content to Markdown
                        markdown_content = markdownify(response.text).strip()

                        # Remove multiple line breaks
                        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

                        return markdown_content

                    except RequestException as e:
                        return f"Error fetching the webpage: {str(e)}"
                    except Exception as e:
                        return f"An unexpected error occurred: {str(e)}"                
                return webpage_content

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
            additional_authorized_imports=AUTHORIZED_IMPORTS
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
        if profile_type not in AGENT_PROFILES:
            return False
            
        dependencies = AGENT_PROFILES[profile_type].get('dependencies', [])
        for dep in dependencies:
            if not any(agent.name == dep for agent in self.active_agents.values()):
                return False
        return True

    def _update_dependencies_graph(self, profile_type: str):
        """Обновляет граф зависимостей при создании нового агента
        
        Args:
            profile_type: Тип профиля созданного агента
        """
        dependencies = AGENT_PROFILES[profile_type].get('dependencies', [])
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
        if profile_type not in AGENT_PROFILES:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        profile = AGENT_PROFILES[profile_type]
        tools = [self._create_tool(tool) for tool in profile['tools']]
        
        if any(tool is None for tool in tools):
            print(f"ВНИМАНИЕ: Не все инструменты были созданы для профиля {profile_type}")
            print(f"Инструменты: {tools}")
        
        # Создаем агента с name и description для возможности управления им
        agent = CodeAgent(
            tools=tools,
            model=profile['model'],
            max_steps=20,
            name=profile_type,
            planning_interval=4 if profile_type == 'researcher' else None,
            description=profile['description'],
            additional_authorized_imports=AUTHORIZED_IMPORTS,
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
        return AGENT_PROFILES[profile_type].get('dependencies', [])
