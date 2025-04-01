# show_me_diagrams.py
#Этот плагин добавляет следующую функциональность:
#Поддерживает генерацию различных типов диаграмм:
#Диаграмма Ганта
#Майнд-карта
#Блок-схема
#Timeline проекта
#Инфографика
#Организационная структура
#Диаграмма процесса
#Создает временный файл с изображением диаграммы
#Возвращает диаграмму как изображение в чат

#Примеры вызова:

#Show Me Diagrams: Create a Gantt chart for a software development project with 5 milestones
#Show Me Diagrams: Generate a mind map for learning machine learning
#Show Me Diagrams: Design a flowchart for customer support process

import os
import tempfile
from typing import Dict, List
import uuid
import subprocess
import logging
from pathlib import Path

from .plugin import Plugin

class ShowMeDiagramsPlugin(Plugin):
    """
    Плагин для генерации различных диаграмм и визуализаций с использованием PlantUML
    """
    def __init__(self):
        self.diagram_types = {
            'gantt_chart': self._generate_gantt_chart,
            'mind_map': self._generate_mind_map,
            'flowchart': self._generate_flowchart, 
            'project_timeline': self._generate_project_timeline,
            'infographic': self._generate_infographic,
            'org_chart': self._generate_org_chart,
            'process_diagram': self._generate_process_diagram,
        }
        # Путь к JAR файлу PlantUML
        self.plantuml_jar = str(Path(__file__).parent / 'plantuml.jar')

        # Промпты для различных типов диаграмм
        self.diagram_prompts = {
            'gantt_chart': """Создай PlantUML код для диаграммы Ганта.
Используй следующие элементы:
- @startgantt/@endgantt
- Project starts
- Задачи в формате [Task] as [T1] lasts X days
- Связи между задачами
- Цветовое оформление
Описание: {description}
Заголовок: {title}""",

            'mind_map': """Создай PlantUML код для интеллект-карты (mind map).
Используй следующие элементы:
- @startmindmap/@endmindmap
- Центральный узел с *
- Ветви с разным количеством *
- Цветовое оформление
Описание: {description}
Заголовок: {title}""",

            'flowchart': """Создай PlantUML код для блок-схемы.
Используй следующие элементы:
- @startuml/@enduml
- start/stop
- if/then/else при необходимости
- Действия в формате :action;
- Стрелки --> для связей
Описание: {description}
Заголовок: {title}""",

            'project_timeline': """Создай PlantUML код для временной шкалы проекта.
Используй следующие элементы:
- @startuml/@enduml
- Временные метки @0, @5 и т.д.
- Описания событий
- Стилизация и цвета
Описание: {description}
Заголовок: {title}""",

            'infographic': """Создай PlantUML код для инфографики.
Используй следующие элементы:
- @startuml/@enduml
- rectangle для блоков
- Вложенные элементы
- Цветовое оформление
- Разные формы (rectangle, circle, artifact)
Описание: {description}
Заголовок: {title}""",

            'org_chart': """Создай PlantUML код для организационной диаграммы.
Используй следующие элементы:
- @startuml/@enduml
- Связи между элементами -->
- Стилизация узлов
- Группировка при необходимости
Описание: {description}
Заголовок: {title}""",

            'process_diagram': """Создай PlantUML код для диаграммы процесса.
Используй следующие элементы:
- @startuml/@enduml
- start/stop
- Действия в формате :action;
- Условия при необходимости
- Параллельные процессы при необходимости
Описание: {description}
Заголовок: {title}""",
        }

    def get_source_name(self) -> str:
        return "Show Me Diagrams"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "generate_diagram",
            "description": "Generate visual diagrams and charts (gantt, flowchart, infographic, mind map, project timeline, process diagram, org chart) from textual descriptions",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string", 
                        "enum": list(self.diagram_types.keys()),
                        "description": "Type of diagram to generate"
                    },
                    "description": {
                        "type": "string", 
                        "description": "Detailed textual description of the diagram contents"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the diagram"
                    }
                },
                "required": ["type", "description"]
            }
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        diagram_type = kwargs.get('type')
        description = kwargs.get('description')
        title = kwargs.get('title', 'Diagram')
        user_id = kwargs.get('user_id')
        if not diagram_type:
            type_prompt = (
                "Выберите тип диаграммы из следующих:\n"
                f"{', '.join(self.diagram_types.keys())}\n"
                "Какой тип диаграммы вы хотите создать?"
            )
            diagram_type_response, _ = await helper.get_chat_response(
                chat_id=helper.conversations.get('last_chat_id', 0),
                query=type_prompt
            )
            diagram_type = diagram_type_response.strip().lower()

            if diagram_type not in self.diagram_types:
                return {"result": f"Неподдерживаемый тип диаграммы: {diagram_type}"}

        if not description:
            description_prompt = f"Пожалуйста, предоставьте подробное описание для диаграммы типа '{diagram_type}'. "
            
            type_hints = {
                'gantt_chart': "Например: Разработка проекта, Дизайн (2 недели), Backend (3 недели), Frontend (4 недели)",
                'mind_map': "Например: Изучение машинного обучения, Математика, Статистика, Программирование, Нейронные сети",
                'flowchart': "Например: Процесс регистрации пользователя -> Ввод email -> Проверка почты -> Создание пароля -> Подтверждение",
                'project_timeline': "Например: Старт проекта, Первый прототип, Бета-тестирование, Релиз, Послерелизная поддержка",
                'infographic': "Например: Статистика продаж, Количество клиентов, Рост выручки, Доля рынка",
                'org_chart': "Например: CEO -> Директор по технологиям -> Руководитель разработки -> Разработчики",
                'process_diagram': "Например: Получение заказа >> Обработка >> Производство >> Доставка >> Получение feedback"
            }

            hint = type_hints.get(diagram_type, "Пожалуйста, опишите содержание диаграммы максимально подробно.")
            description_prompt += hint

            description_response, _ = await helper.get_chat_response(
                chat_id=helper.conversations.get('last_chat_id', 0),
                query=description_prompt
            )
            description = description_response.strip()

        try:
            generator_method = self.diagram_types[diagram_type]
            puml_content, output_file = await generator_method(title, description, helper, user_id)
            if "```" not in puml_content:
                puml_content = "Текст для диаграммы в формате PlantUML:\n```plantuml\n" + puml_content + "\n```"
            return {
                "direct_result": {
                    "kind": "photo",
                    "format": "path",
                    "value": output_file,
                    "add_value": puml_content
                }
            }
        except Exception as e:
            return {"result": f"Error generating diagram: {str(e)}"}
    
    def _generate_plantuml(self, puml_content: str, helper, user_id: int) -> str:
        """Генерирует изображение из PlantUML кода"""
        temp_dir = tempfile.gettempdir()
        file_name = f'diagram_{uuid.uuid4()}'
        puml_file = os.path.join(temp_dir, f'{file_name}.puml')
        output_file = os.path.join(temp_dir, f'{file_name}.png')
        
        # Записываем PlantUML код во временный файл
        with open(puml_file, 'w', encoding='utf-8') as f:
            f.write(puml_content)
        
        # Запускаем PlantUML для генерации изображения
        result = subprocess.run(['java', '-jar', self.plantuml_jar, '-tpng', puml_file, '-o', temp_dir], 
                              capture_output=True, text=True, check=False)
        
        # Пытаемся исправить ошибки до 3 раз
        attempts = 0
        max_attempts = 3
        
        while result.returncode != 0 and attempts < max_attempts:
            attempts += 1
            logging.error(f"Ошибка при генерации PlantUML изображения для {file_name} (попытка {attempts}/{max_attempts}): "
                        f"{result.stderr}")
            print(f"Ошибка при генерации PlantUML (попытка {attempts}/{max_attempts}): {result.stderr}")
            
            # Читаем текущее содержимое файла
            with open(puml_file, 'r', encoding='utf-8') as f:
                current_puml_code = f.read()
                
            messages = [
                {"role": "system", "content": "Ты самый лучший специалист по генерации и исправлению ошибок в коде для PlantUML."},
                {"role": "user", "content": f"При генерации диаграммы возникла ошибка:\n{result.stderr}\n\nВот текущий код PlantUML:\n\n{current_puml_code}\n\nИсправь ошибку, так же проверь код на отсутствие других ошибок и верни правильный код для PlantUML. Верни только исправленный код, без комментариев и объяснений."}
            ]
            response, _ = helper.ask(messages, user_id=user_id)
            
            # Удаляем маркеры кода, если они есть
            generated_text = response.replace("```plantuml", "").replace("```", "").strip()
            
            # Записываем исправленный код обратно в файл
            with open(puml_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            
            # Заново запускаем PlantUML для генерации изображения
            result = subprocess.run(['java', '-jar', self.plantuml_jar, '-tpng', puml_file, '-o', temp_dir], 
                                  capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                logging.info(f"Ошибка исправлена с {attempts} попытки для {file_name}")
                print(f"Ошибка в PlantUML исправлена с {attempts} попытки")
                break
        
        if result.returncode != 0 and attempts >= max_attempts:
            logging.error(f"Не удалось исправить ошибки в PlantUML после {max_attempts} попыток для {file_name}")
            print(f"Не удалось исправить ошибки в PlantUML после {max_attempts} попыток")
            raise Exception(f"Не удалось сгенерировать диаграмму после {max_attempts} попыток")
        
        # Проверяем, что файл был создан
        if not os.path.exists(output_file):
            raise Exception(f"Файл диаграммы не был создан: {output_file}")
        
        # Удаляем временный файл с кодом
        try:
            os.remove(puml_file)
        except Exception as e:
            logging.warning(f"Не удалось удалить временный файл {puml_file}: {str(e)}")
        
        return puml_content, output_file

    async def _generate_diagram_code(self, diagram_type: str, title: str, description: str, helper, user_id: int) -> str:
        """Генерирует PlantUML код с помощью ask"""
        prompt = self.diagram_prompts[diagram_type].format(
            description=description,
            title=title
        )
        plantuml_code, _ = await helper.ask(
            prompt=prompt,
            user_id=user_id,
            assistant_prompt="Ты помощник, который создает PlantUML код для диаграмм. Лучше тебя в этом не разбирается никто! Ты должен использовать все свои знания и навыки для того, чтобы помочь пользователю. Всегда используй цвета, стили, формы, размеры, шрифты, чтобы сделать диаграмму более наглядной, размещай код офомления в начале. Так же используй Creole синтаксис, OpenIconic, Emoji, чтобы сделать диаграмму более наглядной. Вернуть нужно только PlantUML код, никаких комментариев и объяснений, без лишних ковычек, это очень важно! Всегда проверяй сгенерированный код на ошибки и неточности, если есть ошибки, исправь их."
        )
        return plantuml_code.strip()

    async def _generate_gantt_chart(self, title: str, description: str, helper, user_id: int) -> str:
        puml_code = await self._generate_diagram_code('gantt_chart', title, description, helper, user_id)
        return self._generate_plantuml(puml_code, helper, user_id)

    async def _generate_flowchart(self, title: str, description: str, helper, user_id: int) -> str:
        puml_code = await self._generate_diagram_code('flowchart', title, description, helper, user_id)
        return self._generate_plantuml(puml_code, helper, user_id)

    async def _generate_mind_map(self, title: str, description: str, helper, user_id: int) -> str:
        puml_code = await self._generate_diagram_code('mind_map', title, description, helper, user_id)
        return self._generate_plantuml(puml_code, helper, user_id)

    async def _generate_project_timeline(self, title: str, description: str, helper, user_id: int) -> str:
        puml_code = await self._generate_diagram_code('project_timeline', title, description, helper, user_id)
        return self._generate_plantuml(puml_code, helper, user_id)

    async def _generate_infographic(self, title: str, description: str, helper, user_id: int) -> str:
        puml_code = await self._generate_diagram_code('infographic', title, description, helper, user_id)
        return self._generate_plantuml(puml_code, helper, user_id)

    async def _generate_org_chart(self, title: str, description: str, helper, user_id: int) -> str:
        puml_code = await self._generate_diagram_code('org_chart', title, description, helper, user_id)
        return self._generate_plantuml(puml_code, helper, user_id)

    async def _generate_process_diagram(self, title: str, description: str, helper, user_id: int) -> str:
        puml_code = await self._generate_diagram_code('process_diagram', title, description, helper, user_id)
        return self._generate_plantuml(puml_code, helper, user_id)