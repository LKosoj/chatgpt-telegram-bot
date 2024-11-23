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
import json
import random
import tempfile
from typing import Dict
from PIL import Image, ImageDraw, ImageFont
import uuid
import math  # Добавлен импорт math

from .plugin import Plugin

class ShowMeDiagramsPlugin(Plugin):
    """
    A plugin to generate various diagrams and visualizations
    """
    def __init__(self):
        self.diagram_types = {
            'gantt_chart': self._generate_gantt_chart,
            'mind_map': self._generate_mind_map,
            'flowchart': self._generate_flowchart, 
            'project_timeline': self._generate_project_timeline,
            'infographic': self._generate_infographic,
            'org_chart': self._generate_org_chart,
            'process_diagram': self._generate_process_diagram
        }

    def get_source_name(self) -> str:
        return "Show Me Diagrams"

    def get_spec(self) -> [Dict]:
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

        # Проверяем наличие необходимой информации
        if not diagram_type:
            # Запрос типа диаграммы
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

            # Проверка корректности выбранного типа
            if diagram_type not in self.diagram_types:
                return {"result": f"Unsupported diagram type: {diagram_type}"}

        if not description:
            # Запрос описания для конкретного типа диаграммы
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
            output_file = generator_method(title, description)
            return {
                "direct_result": {
                    "kind": "photo",
                    "format": "path",
                    "value": output_file
                }
            }
        except Exception as e:
            return {"result": f"Error generating diagram: {str(e)}"}
    
    def _generate_gantt_chart(self, title: str, description: str) -> str:
        """Mock Gantt Chart generation"""
        img = Image.new('RGB', (800, 400), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        d.text((20, 20), title, fill='black', font=font)
        d.text((20, 50), "Gantt Chart: " + description, fill='black', font=font)

        # Mock task visualization
        tasks = description.split(',')
        for i, task in enumerate(tasks):
            x1 = 50 + i * 100
            x2 = x1 + random.randint(50, 150)
            d.rectangle([x1, 150 + i*30, x2, 170 + i*30], fill='blue', outline='black')
            d.text((x1, 180 + i*30), task, fill='black', font=font)

        return self._save_temp_image(img)

    def _generate_flowchart(self, title: str, description: str) -> str:
        """Mock Flowchart generation"""
        steps = description.split('->')
        if not steps:
            steps = ["No steps defined"]

        img = Image.new('RGB', (800, 600), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        d.text((20, 20), title, fill='black', font=font)
        d.text((20, 50), "Flowchart: " + description, fill='black', font=font)

        for i, step in enumerate(steps):
            x = 200
            y = 150 + i * 100
            d.rectangle([x, y, x+400, y+50], outline='black')
            d.text((x+10, y+10), step.strip(), fill='black', font=font)
            
            if i < len(steps) - 1:
                d.line([(x+200, y+50), (x+200, y+100)], fill='black', width=2)
                d.polygon([(x+195, y+100), (x+205, y+100), (x+200, y+110)], fill='black')

        return self._save_temp_image(img)

    def _generate_mind_map(self, title: str, description: str) -> str:
        """Mock Mind Map generation"""
        branches = description.split(',')
        if not branches:
            branches = ["No branches defined"]

        img = Image.new('RGB', (800, 600), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        d.text((20, 20), title, fill='black', font=font)
        d.text((20, 50), "Mind Map: " + description, fill='black', font=font)

        # Central concept
        d.ellipse([350, 250, 450, 350], fill='lightblue', outline='black')
        d.text((370, 280), "Central\nIdea", fill='black', font=font)

        # More mathematically precise branch generation
        num_branches = len(branches)
        for i, branch in enumerate(branches):
            angle = 2 * math.pi * i / num_branches
            x = 400 + int(200 * math.cos(angle))
            y = 300 + int(200 * math.sin(angle))
            
            d.line([(400, 300), (x, y)], fill='green', width=2)
            d.text((x, y), branch.strip(), fill='black', font=font)

        return self._save_temp_image(img)

    def _generate_project_timeline(self, title: str, description: str) -> str:
        """Mock Project Timeline generation"""
        img = Image.new('RGB', (800, 400), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        d.text((20, 20), title, fill='black', font=font)
        d.text((20, 50), "Project Timeline: " + description, fill='black', font=font)

        milestones = description.split(',')
        for i, milestone in enumerate(milestones):
            x = 100 + i * 150
            d.ellipse([x-20, 200, x+20, 240], fill='red', outline='black')
            d.text((x-50, 250), milestone, fill='black', font=font)
            d.line([(x, 240), (x, 280)], fill='black', width=2)

        return self._save_temp_image(img)

    def _generate_infographic(self, title: str, description: str) -> str:
        """Mock Infographic generation"""
        img = Image.new('RGB', (800, 600), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        d.text((20, 20), title, fill='black', font=font)
        d.text((20, 50), "Infographic: " + description, fill='black', font=font)

        stats = description.split(',')
        colors = ['red', 'green', 'blue', 'purple']
        for i, stat in enumerate(stats):
            d.rectangle([100, 150 + i*100, 700, 220 + i*100], 
                        fill=colors[i % len(colors)]
                        ,outline='black')
            d.text((120, 170 + i*100), stat, fill='black', font=font)
        return self._save_temp_image(img)

    def _generate_org_chart(self, title: str, description: str) -> str:
        """Mock Organization Chart generation"""
        img = Image.new('RGB', (800, 600), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        d.text((20, 20), title, fill='black', font=font)
        d.text((20, 50), "Organization Chart: " + description, fill='black', font=font)

        roles = description.split('->')
        for i, role in enumerate(roles):
            x = 300
            y = 150 + i * 100
            d.rectangle([x, y, x+200, y+50], outline='black')
            d.text((x+10, y+10), role.strip(), fill='black', font=font)
            
            if i > 0:
                d.line([(x+100, y), (x+100, y-50)], fill='black', width=2)
                d.polygon([(x+95, y), (x+105, y), (x+100, y-10)], fill='black')

        return self._save_temp_image(img)

    def _generate_process_diagram(self, title: str, description: str) -> str:
        """Mock Process Diagram generation"""
        img = Image.new('RGB', (800, 600), color='white')
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        d.text((20, 20), title, fill='black', font=font)
        d.text((20, 50), "Process Diagram: " + description, fill='black', font=font)

        steps = description.split('>>')
        for i, step in enumerate(steps):
            x = 200
            y = 150 + i * 100
            d.ellipse([x, y, x+200, y+50], outline='black')
            d.text((x+10, y+10), step.strip(), fill='black', font=font)
            
            if i < len(steps) - 1:
                d.line([(x+100, y+50), (x+100, y+100)], fill='black', width=2)
                d.polygon([(x+95, y+100), (x+105, y+100), (x+100, y+110)], fill='black')

        return self._save_temp_image(img)

    def _save_temp_image(self, img: Image) -> str:
        """Save image to a temporary file"""
        temp_dir = tempfile.gettempdir()
        filename = os.path.join(temp_dir, f'diagram_{uuid.uuid4()}.png')
        img.save(filename)
        return filename