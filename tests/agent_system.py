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
from agent_factory import AgentFactory, model_lite, model_hard, model_search

class DynamicAgentSystem:
    """Система с динамическим созданием и управлением агентами"""
    
    def __init__(self):
        self.factory = AgentFactory()
        self.task_queue = asyncio.Queue()
        self.agent_pool = {}
        # Для хранения промежуточных результатов
        self.shared_results = {}
    
    def get_agent_dependencies(self, agent_type: str) -> List[str]:
        """Получает список зависимостей для агента из его профиля"""
        return self.factory.AGENT_PROFILES[agent_type].get('dependencies', [])
    
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
        for agent_type, profile in self.factory.AGENT_PROFILES.items():
            agents_info[agent_type] = {
                'description': profile.get('description', 'Описание отсутствует'),
                'dependencies': profile.get('dependencies', []),
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
                    diagram_description += f"\nЗависимости: {', '.join(info['dependencies'])}"
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
            analysis_prompt = f"""
            Определи какие типы агентов нужны для выполнения задачи. 
            Доступные типы, их описание и зависимости:
            {', '.join(f"{k} ({v['description']}) - {', '.join(v['dependencies'])}" for k, v in self.factory.AGENT_PROFILES.items())}
            
            Задача: {task}
            
            Верни только список типов через запятую, без кавычек.
            Выбирай только тех агентов, которые явно требуются для выполнения задачи.
            """
            
            model = model_lite
            messages = [
                {"role": "system", "content": "Ты помощник, который анализирует задачи и определяет необходимые типы агентов. Возвращай типы агентов без кавычек. Выбирай только тех агентов, которые явно требуются для выполнения задачи."},
                {"role": "user", "content": analysis_prompt}
            ]
            response = model(messages)
            
            if not response.content.strip():
                raise ValueError("Получен пустой ответ от модели")
                
            agent_types = [a.strip().strip("'\"") for a in response.content.split(',')]
            invalid_types = [t for t in agent_types if t not in self.factory.AGENT_PROFILES]
            if invalid_types:
                raise ValueError(f"Обнаружены недопустимые типы агентов: {invalid_types}")
                
            # Добавляем только необходимые зависимости для выбранных агентов
            all_required_agents = set(agent_types)
            for agent_type in agent_types:
                dependencies = self.get_agent_dependencies(agent_type)
                # Добавляем только прямые зависимости
                all_required_agents.update(dependencies)
            
            return list(all_required_agents)
        except Exception as e:
            print(f"Ошибка при анализе задачи: {str(e)}")
            return ['researcher']

    def can_start_agent(self, agent_type: str) -> bool:
        """Проверяет, готовы ли все зависимости для запуска агента"""
        dependencies = self.get_agent_dependencies(agent_type)
        if not dependencies:
            return True
        
        for dependency in dependencies:
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
                    if dep in self.shared_results and self.shared_results[dep]:
                        context += f"\nРезультаты от {dep}:\n{self.shared_results[dep]}\n"
            
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
                        'subtask': f"{initial_task}\n\nВаша роль - {self.factory.AGENT_PROFILES[agent_type]['description']}"
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
                if info['results']:
                    report.append(f"📋 Результаты агента {agent_id}:")
                    for idx, result in enumerate(info['results'], 1):
                        report.append(f"  Результат #{idx}:")
                        try:
                            parsed_result = json.loads(result)
                            for key, value in parsed_result.items():
                                report.append(f"    {key}: {value}")
                        except:
                            report.append(f"    {result}")
                    report.append("")
            
            return "\n".join(report)
                    
        except Exception as e:
            print(f"Критическая ошибка в координации: {str(e)}")
            return f"Ошибка: {str(e)}"

def show_available_agents(system: DynamicAgentSystem):
    # Выводим список доступных агентов
    print("\n📋 Доступные агенты:")
    print("=" * 50)
    agents, diagram = system.get_available_agents()
    for agent_type, info in agents.items():
        print(f"\n🤖 {agent_type}:")
        print(f"   📝 Описание: {info['description']}")
        if info['dependencies']:
            print(f"   🔗 Зависимости: {', '.join(info['dependencies'])}")
        if info['capabilities']:
            print(f"   💪 Возможности: {', '.join(info['capabilities'])}")
    print("=" * 50 + "\n")
    print(f"Диаграмма агентов:\n{diagram}")
    print("=" * 50 + "\n")
            

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
    #show_available_agents(system)
    #return

    # Примеры для тестирования мультиагентной системы. Запускать по одному, остальные комментировать!!!
    complex_task = analyze_ai_trends(system)
    #complex_task = analyze_real_estate_trends(system)
    #complex_task = analyze_data_trends(system)
    #complex_task = analyze_crypto_trends(system)
    #complex_task = analyze_crypto_system(system)
    #complex_task = create_mind_map(system)
    

    content = await system.coordinate(complex_task)
    
    print("\n" + "=" * 50)
    print(content)
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())