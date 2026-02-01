from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, tool, OpenAIServerModel
from typing import Dict, Any, List, Optional
import os
import asyncio
import re
import matplotlib
import json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import logging
import uuid
import subprocess
import tempfile
from pathlib import Path
from agent_factory import AgentFactory, model_lite, model_hard, model_search, AGENT_PROFILES
from datetime import datetime
class DynamicAgentSystem:
    """Система с динамическим созданием и управлением агентами"""
    
    def __init__(self):
        self.factory = AgentFactory()
        self.task_queue = asyncio.Queue()
        self.agent_pool = {}
        # Для хранения промежуточных результатов
        self.shared_results = {}
        # Путь к JAR файлу PlantUML
        self.plantuml_jar = str(Path(__file__).parent / 'plantuml.jar')

    def get_agent_dependencies(self, agent_type: str) -> List[str]:
        """Получает список зависимостей для агента из его профиля"""
        return AGENT_PROFILES[agent_type].get('dependencies', [])
    
    def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает словарь всех доступных агентов с их описаниями, зависимостями и возможностями
        
        Returns:
            Dict[str, Dict[str, Any]]: Словарь, где ключ - тип агента, значение - словарь с информацией об агенте:
                - description (str): Описание агента
                - dependencies (List[str]): Список зависимых агентов
                - capabilities (List[str]): Список конкретных возможностей агента
                - tools (List[str]): Список доступных инструментов
                - api_integrations (List[str]): Список интеграций с внешними API
        """
        agents_info = {}
        diagram_result = None
        for agent_type, profile in AGENT_PROFILES.items():
            # Преобразуем зависимости в читаемый формат
            formatted_dependencies = []
            for dep in profile.get('dependencies', []):
                if isinstance(dep, list):
                    # Если зависимость - список альтернатив, объединяем через " или "
                    formatted_dependencies.append(' или '.join(dep))
                else:
                    formatted_dependencies.append(dep)
            
            agents_info[agent_type] = {
                'description': profile.get('description', 'Описание отсутствует').split('\n')[0],
                'dependencies': formatted_dependencies,
                'capabilities': profile.get('capabilities', []),
                'tools': profile.get('tools', []),
                'api_integrations': profile.get('api_integrations', [])
            }
        
        # Создаем диаграмму агентов
        try:
            diagram_agent = self.factory.create_agent('diagram_creator')
            diagram_description = """
            Создай диаграмму системы агентов со следующей информацией:
            """
            for agent_type, info in agents_info.items():
                diagram_description += f"\n\nАгент: {agent_type}"
                diagram_description += f"\nОписание: {info['description']}"
                if info['dependencies']:
                    # Преобразуем зависимости в строку, учитывая возможные альтернативы
                    dependencies_str = ', '.join(str(dep) for dep in info['dependencies'])
                    diagram_description += f"\nЗависимости: {dependencies_str}"
                if info['capabilities']:
                    diagram_description += f"\nВозможности: {', '.join(info['capabilities'])}"
            
            diagram_result = diagram_agent.run(diagram_description)
            if isinstance(diagram_result, str):
                print("\n🎨 Диаграмма агентов создана и сохранена")
        except Exception as e:
            print(f"\n⚠️ Не удалось создать диаграмму агентов: {str(e)}")
        
        return agents_info, diagram_result
    
    async def analyze_task(self, task: str) -> List[str]:
        """Анализ задачи и определение необходимых агентов"""
        try:
            dependencies = {', '.join((' ИЛИ '.join(dep) if isinstance(dep, list) else dep) for dep in v['dependencies']) for k, v in AGENT_PROFILES.items()}
            print(dependencies)
            analysis_prompt = f"""
            Определи какие типы агентов нужны для выполнения задачи. 
            Доступные типы, их описание и зависимости.
            Зависимости: {', '.join(f"{k} ({v['description']}) - {dependencies}" for k, v in AGENT_PROFILES.items())}
            Добавляй зависимости, только если они нужны для выполнения задачи! Лишних не добавляй, это очень важно, иначе результат будет неверным и пользователь расстроится):
            
            Задача: {task}
            
            Верни только список типов через запятую, без кавычек.
            Выбирай только тех агентов, которые явно требуются для выполнения задачи.
            """
            
            model = model_lite
            messages = [
                {"role": "system", "content": "Ты помощник, который анализирует задачи и определяет необходимые типы агентов. Возвращай типы агентов без кавычек. Выбирай только тех агентов, которые явно требуются для выполнения задачи. Если типы указаны через ИЛИ, значит можно выбрать одного из них, если второй не нужен, не добавляй лишних агентов!"},
                {"role": "user", "content": analysis_prompt}
            ]
            response = model(messages)
            
            if not response.content.strip():
                raise ValueError("Получен пустой ответ от модели")
                
            agent_types = [a.strip().strip("'\"") for a in response.content.split(',')]
            invalid_types = [t for t in agent_types if t not in AGENT_PROFILES]
            if invalid_types:
                raise ValueError(f"Обнаружены недопустимые типы агентов: {invalid_types}")
                
            # Добавляем только необходимые зависимости для выбранных агентов
            all_required_agents = set(agent_types)
            for agent_type in agent_types:
                dependencies = self.get_agent_dependencies(agent_type)
                # Добавляем только прямые зависимости, учитывая альтернативы
                for dep in dependencies:
                    if isinstance(dep, list):
                        all_required_agents.update(dep)
                    else:
                        all_required_agents.add(dep)
            
            return list(all_required_agents)
        except Exception as e:
            print(f"Ошибка при анализе задачи: {str(e)}")
            return ['researcher']


    def can_start_agent(self, agent_type: str) -> bool:
        """Проверяет, готовы ли зависимости для запуска агента с поддержкой альтернативных зависимостей."""
        dependencies = self.get_agent_dependencies(agent_type)
        if not dependencies:
            return True
        for dependency in dependencies:
            # Если зависимость задана как список альтернатив
            if isinstance(dependency, list):
                if not any(dep in self.shared_results and self.shared_results[dep] for dep in dependency):
                    return False
            else:
                if dependency not in self.shared_results or not self.shared_results[dependency]:
                    return False
        return True

    async def assign_task(self, agent: CodeAgent, task: str):
        """Назначение задачи агенту и обработка результата"""
        try:
            # Получаем agent_id и тип агента
            agent_id = getattr(agent, 'agent_id', 'unknown')
            agent_type = agent_id.split('-')[0]
            
            # Формируем контекст с результатами зависимостей
            context = ""
            dependencies = self.get_agent_dependencies(agent_type)
            if dependencies:
                context = "\nКонтекст от других агентов:\n"
                for dep in dependencies:
                    # Если зависимость - список альтернатив, берем первый непустой элемент
                    dep0 = None
                    if isinstance(dep, list):
                        dep0 = next((d for d in dep if d in self.shared_results and self.shared_results[d]), None)
                        if dep0 is None:
                            dep0 = dep[0]  # Если все элементы пусты, берем первый
                    else:
                        dep0 = dep
                    
                    if dep0 in self.shared_results and self.shared_results[dep0]:
                        context += f"\nРезультаты от {dep0}:\n{self.shared_results[dep0]}\n"
            
            # Добавляем контекст к задаче
            task_with_context = f"{task}\n{context}" if context else task
            
            # Запускаем агента
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, agent.run, task_with_context)
            
            # Сохраняем результат в общем хранилище
            self.shared_results[agent_type] = result
            
            # Обновляем информацию агента
            for info in self.agent_pool.values():
                if info['agent'] == agent:
                    info['results'].append(result)
                    info['status'] = 'idle'
                    break
                
        except Exception as e:
            error_msg = f"Ошибка в агенте {agent_id}: {str(e)}"
            print(error_msg)
            for info in self.agent_pool.values():
                if info['agent'] == agent:
                    info['status'] = 'idle'
                    info['results'].append(f"[ОШИБКА] {error_msg}")
                    break

    async def coordinate(self, initial_task: str):
        """Координация выполнения задачи"""
        try:
            required_agents = await self.analyze_task(initial_task)
            print(f"Необходимые агенты: {required_agents}")
            if not required_agents:
                print("Не удалось определить необходимых агентов")
                return
            session_id = str(uuid.uuid4())[:8]

            # Создание агентов и подзадач
            # Добавляем исходную задачу к подзадачам
            for agent_type in required_agents:
                agent = self.factory.create_agent(agent_type)
                agent_id = getattr(agent, 'agent_id', f"{agent_type}-unknown")
                if agent_id not in self.agent_pool:
                    self.agent_pool[agent_id] = {
                        'agent': agent,
                        'status': 'idle',
                        'results': [],
                        'subtask': f"Исходная задача: {initial_task}\n\nВаша роль - {AGENT_PROFILES[agent_type]['description']}\n {AGENT_PROFILES[agent_type]['prompt_templates']}. session_id: {session_id}. Текущие дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
            
            # Выполняем агентов с учетом зависимостей
            while True:
                tasks = []
                all_completed = True
                
                for agent_id, info in self.agent_pool.items():
                    agent_type = agent_id.split('-')[0]
                    
                    if info['status'] == 'idle' and not info['results']:
                        all_completed = False
                        if self.can_start_agent(agent_type):
                            info['status'] = 'busy'
                            task = asyncio.create_task(
                                self.assign_task(info['agent'], info['subtask'])
                            )
                            tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks)
                elif all_completed:
                    break
                else:
                    # Ждем немного перед следующей проверкой
                    await asyncio.sleep(0.1)
            
            # Формируем итоговый отчет
            report = []
            report.append("=== ИТОГОВЫЙ ОТЧЕТ ===\n")
            report.append("🔍 Выполненные задачи:")
            report.append(f"- Исходная задача: {initial_task}")
            report.append(f"- Количество агентов: {len(self.agent_pool)}")
            report.append("")
            
            # Добавляем результаты каждого агента
            for agent_id, info in self.agent_pool.items():
                agent_type = agent_id.split('-')[0]
                if agent_type in ['visualizer', 'researcher']:
                    continue
                if agent_type == 'diagram_creator':
                    diagrams = info['results']
                    diagrams_str = '\n'.join(str(diagram) for diagram in diagrams)
                    print(f"Диаграммы:\n{diagrams_str}")
                    _, _ = self._generate_plantuml(diagrams_str, session_id)
                if info['results']:
                    report.append(f"📋 Результаты агента {agent_id}:")
                    for idx, result in enumerate(info['results'], 1):
                        report.append(f"  Результат #{idx}:")
                        try:
                            # Обработка различных типов результатов
                            if isinstance(result, str):
                                # Если результат - строка, пытаемся распарсить как JSON
                                try:
                                    parsed_result = json.loads(result)
                                    for key, value in parsed_result.items():
                                        report.append(f"    {key}: {value}")
                                except (json.JSONDecodeError, TypeError):
                                    # Если не JSON, выводим как есть
                                    report.append(f"    {result}")
                            elif isinstance(result, dict):
                                # Если результат - словарь, выводим его содержимое
                                for key, value in result.items():
                                    report.append(f"    {key}: {value}")
                            elif isinstance(result, list):
                                # Если результат - список, выводим его элементы
                                for item in result:
                                    report.append(f"    {item}")
                            else:
                                # Для остальных типов используем str()
                                report.append(f"    {str(result)}")
                        except Exception as e:
                            report.append(f"    Ошибка при обработке результата: {str(e)}")
                    report.append("")
            self.advanced_visualization(report, session_id)
            return "\n".join(report)
                    
        except Exception as e:
            print(f"Критическая ошибка в координации: {str(e)}")
            return f"Ошибка: {str(e)}"

    def show_available_agents(self):
        # Выводим список доступных агентов
        session_id = str(uuid.uuid4())[:8]
        result = "\n📋 Доступные агенты:"
        result += "=" * 50
        agents, diagram = self.get_available_agents()
        for agent_type, info in agents.items():
            result += f"\n🤖 {agent_type}:"
            result += f"   📝 Описание: {info['description']}"
            if info['dependencies']:
                dependencies = info['dependencies']
                result += f"   🔗 Зависимости: {dependencies}"
            if info['capabilities']:
                result += f"   💪 Возможности: {', '.join(info['capabilities'])}"
        result += "\n" + "=" * 50 + "\n"
        _, _ = self._generate_plantuml(diagram, session_id)
        result += f"Диаграмма агентов:\n{diagram}"
        result += "\n" + "=" * 50 + "\n"
        print(result)
        self.advanced_visualization(result, session_id)

    def advanced_visualization(self, result, session_id):
        """
        Создаёт HTML страницу из сохраненных графиков.
        
        Args:
            output_path (str): Путь для сохранения HTML файла с графиками
        """
        output_path=f"interactive_plots_{session_id}.html"
        try:
            # Проверяем наличие директории с графиками
            plots_dir = 'plots'
            os.makedirs(plots_dir, exist_ok=True)

            plot_files = []
            # Получаем список всех PNG файлов
            if not os.path.exists(plots_dir):
                logging.error("Директория с графиками не найдена")
            else:
                plot_files = [f for f in os.listdir(plots_dir) if f'_{session_id}' in f]
            
            if not plot_files:
                logging.error("Графики не найдены")

            # Создаем HTML страницу
            html_content = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                '    <meta charset="utf-8">',
                '    <title>Визуализация данных</title>',
                '    <style>',
                '        .plot-container {',
                '            max-width: 800px;',
                '            margin: 20px auto;',
                '            padding: 20px;',
                '            border: 1px solid #ddd;',
                '            border-radius: 5px;',
                '        }',
                '        img {',
                '            max-width: 100%;',
                '            height: auto;',
                '            display: block;',
                '            margin: 0 auto;',
                '        }',
                '        h2 {',
                '            text-align: center;',
                '            color: #333;',
                '        }',
                '        pre {',
                '            white-space: pre-wrap;',
                '            word-wrap: break-word;',
                '            background-color: #f5f5f5;',
                '            padding: 15px;',
                '            border-radius: 5px;',
                '            font-family: monospace;',
                '            font-size: 14px;',
                '            line-height: 1.4;',
                '            overflow-x: auto;',
                '        }',
                '    </style>',
                '</head>',
                '<body>'
            ]

            # Добавляем каждый график в HTML
            for i, plot_file in enumerate(sorted(plot_files), 1):
                plot_path = os.path.join(plots_dir, plot_file)
                
                # Конвертируем изображение в base64
                with open(plot_path, 'rb') as img_file:
                    import base64
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                html_content.extend([
                    '    <div class="plot-container">',
                    f'        <h2>График {i}</h2>',
                    f'        <img src="data:image/png;base64,{img_data}" alt="График {i}">',
                    '    </div>'
                ])

            result_str = '\n'.join(result) if isinstance(result, list) else str(result)
            html_content.extend([
                '    <div class="result-container">',
                '        <h2>Результаты анализа</h2>',
                f'        <pre>{result_str}</pre>',
                '    </div>'
            ])

            html_content.append('</body></html>')

            # Сохраняем HTML файл
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))

            logging.info(f"HTML страница с графиками сохранена в {output_path}")
            self.clean_data(session_id)            
        except Exception as e:
            logging.error(f"Ошибка при создании HTML страницы: {e}")
        finally:
            self.clean_data(session_id)
            pass

    def clean_data(self, session_id):
        """Удаляет файлы с суффиксом _{session_id} в каталогах data и plots."""
        for file in os.listdir('data'):
            if f'_{session_id}' in file:
                os.remove(os.path.join('data', file))
        for file in os.listdir('plots'):
            if f'_{session_id}' in file:
                os.remove(os.path.join('plots', file))

    def _generate_plantuml(self, puml_content: str, session_id: str) -> str:
        """Генерирует изображение из PlantUML кода"""
        temp_dir = tempfile.gettempdir()
        file_name = f'diagram_{session_id}'
        puml_file = os.path.join('plots', f'{file_name}.puml')
        output_file = os.path.join('plots', f'{file_name}.png')
        
        # Записываем PlantUML код во временный файл
        with open(puml_file, 'w', encoding='utf-8') as f:
            f.write(puml_content)
        
        # Запускаем PlantUML для генерации изображения
        result = subprocess.run(['java', '-jar', self.plantuml_jar, '-tpng', puml_file ])
        
        # Удаляем временный файл с кодом
        os.remove(puml_file)
        
        return puml_content, output_file

def analyze_ai_trends(system: DynamicAgentSystem):
    complex_task = """
    Проанализировать последние тенденции в области ИИ за текущий год:
    1. Собрать данные о новых исследованиях
    2. Сравнить с предыдущими годами
    3. Создать визуализацию основных трендов
    4. Проверить достоверность источников
    5. Подготовить итоговый отчет на русском языке
    """
    return complex_task

def analyze_real_estate_trends(system: DynamicAgentSystem):
    complex_task = """
    Проанализировать динамику цен на жилье в Москве за последние 5 лет.
    Сравнить с инфляцией и доходностью основных инвестиционных инструментов.
    Создать визуализацию основных трендов.
    Подготовить итоговый отчет для инвесторов на русском языке.
    """
    return complex_task

def analyze_data_trends(system: DynamicAgentSystem):
    complex_task = """
    Проанализировать тренды в области работы с данными для построения аналитических платформ. Использовать все доступные источники.
    Предоставить список трендов и их описание.
    Подготовить итоговый отчет.
    """
    return complex_task

def analyze_crypto_trends(system: DynamicAgentSystem):
    complex_task = """
    Проанализировать тренды в области криптовалют.
    Предоставить список трендов и их описание.
    Построить диаграмму трендов.
    Подготовить итоговый отчет.
    """
    return complex_task

def analyze_crypto_system(system: DynamicAgentSystem):
    complex_task = """
    Построить архитектуру системы для анализа и прогнозирования цен на криптовалюту.
    Использовать все доступные источники.
    Предоставить список компонентов и их описание.
    Нарисовать диаграмму компонентов.
    """
    return complex_task

def create_mind_map(system: DynamicAgentSystem):
    complex_task = """
    Подготовить информацию для построения mind map по теме "Искусственный интеллект - Использование Агентов".
    Использовать все доступные источники.
    Предоставить список компонентов и их описание.
    Нарисовать диаграмму компонентов.
    """ 
    return complex_task 

async def main():
    system = DynamicAgentSystem()
    
    # Показывает доступных агентов и их диаграмму зависимостей
    #system.show_available_agents()
    #return

    # Примеры для тестирования мультиагентной системы. Запускать по одному, остальные комментировать!!!
    #complex_task = analyze_ai_trends(system)
    #complex_task = analyze_real_estate_trends(system)
    #complex_task = analyze_data_trends(system)
    #complex_task = analyze_crypto_trends(system)
    complex_task = analyze_crypto_system(system)
    #complex_task = create_mind_map(system)
    

    content = await system.coordinate(complex_task)
    
    print("\n" + "=" * 50)
    print(content)
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())