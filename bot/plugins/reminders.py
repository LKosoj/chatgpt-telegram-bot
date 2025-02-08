import logging
from typing import Dict, Any, List
from plugins.plugin import Plugin
from datetime import datetime
import json
import os
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

class RemindersPlugin(Plugin):
    """
    Плагин для управления напоминаниями и интеграциями
    """
    def __init__(self):
        self.reminders = {}
        self.reminders_file = os.path.join(os.path.dirname(__file__), "reminders.json")
        self.load_reminders()

    def get_source_name(self) -> str:
        return "Reminders"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "set_reminder",
            "description": f'Установить напоминание на определенное время, сейчас {datetime.now().strftime("%Y%m%d%H%M%S")}',
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Время напоминания в формате YYYY-MM-DD HH:MM"
                    },
                    "message": {
                        "type": "string",
                        "description": "Текст напоминания"
                    },
                    "current_time": {
                        "type": "string",
                        "description": f'Текущее время {datetime.now().strftime("%Y-%m-%d %H:%M")}'
                    },
                    "integration": {
                        "type": "string", 
                        "description": "Интеграция для отправки (telegram)",
                        "enum": ["telegram",]
                    }
                },
                "required": ["time", "message", "integration","current_time"]
            }
        },
        {
            "name": "list_reminders",
            "description": "Показать список активных напоминаний",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_time": {
                        "type": "string",
                        "description": f'Текущее время {datetime.now().strftime("%Y-%m-%d %H:%M")}'
                    }
                },
            }
        },
        {
            "name": "delete_reminder", 
            "description": "Удалить напоминание по ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "string",
                        "description": "ID напоминания для удаления"
                    }
                },
                "required": ["reminder_id"]
            }
        }]

    def get_commands(self) -> List[Dict]:
        """Возвращает список команд, которые поддерживает плагин"""
        return [
            {
                "command": "set_reminder",
                "description": "Установить напоминание на определенное время",
                "handler": self.execute,
                "handler_kwargs": {"function_name": "set_reminder"},
                "args": "<time> <message>",
                "plugin_name": "reminders",
            },
            {
                "command": "list_reminders",
                "description": "Показать список активных напоминаний",
                "handler": self.handle_prompt_constructor,
                "handler_kwargs": {},
                "plugin_name": "reminders",
                "add_to_menu": True,
            },
            {
                "command": "delete_reminder",
                "description": "Удалить напоминание",
                "args": "<reminder_id>",
                "handler": self.execute,
                "handler_kwargs": {"function_name": "delete_reminder"},
                "plugin_name": "reminders"
            },
            {
                # Обработчик для всех callback_query плагина
                "callback_query_handler": self.handle_reminder_callback,
                "callback_pattern": "^reminder:",
                "plugin_name": "reminders",
                "handler_kwargs": {}
            }
        ]

    async def handle_prompt_constructor(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик для конструктора промптов"""
        self.load_reminders()
        message = update.message
        user_id = str(message.from_user.id) if message else None
        user_reminders = self.reminders.get(user_id, {})

        if not user_reminders:
            await message.reply_text(
                "У вас нет активных напоминаний.",
                parse_mode='Markdown'
            )
            return               
        
        # Создаем inline кнопки для каждого напоминания
        keyboard = []
        
        for r_id, r in user_reminders.items():            
            reminder_time = datetime.fromisoformat(r['time'])
            formatted_time = reminder_time.strftime('%d.%m.%Y %H:%M')
            
            # Создаем ряд из двух кнопок для каждого напоминания
            keyboard.append([
                InlineKeyboardButton(
                    text=f"📅 {formatted_time} 📝 {r['message']}",
                    callback_data=f"reminder:view:{r_id}"
                ),
                InlineKeyboardButton(
                    text="❌ Удалить",
                    callback_data=f"reminder:delete:{r_id}"
                )
            ])
        
        # Добавляем кнопку закрытия
        keyboard.append([
            InlineKeyboardButton("❌ Закрыть", callback_data="reminder:close_menu:")
        ])

        # Создаем разметку с кнопками
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await message.reply_text(
            "Ваши напоминания:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    def load_reminders(self) -> None:
        """Загружает напоминания из файла хранения"""
        try:
            if os.path.exists(self.reminders_file):
                with open(self.reminders_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.reminders = data
                    else:
                        self.reminders = {}
            else:
                self.reminders = {}
        except Exception as e:
            logging.exception(f"Ошибка при загрузке напоминаний: {e}")
            self.reminders = {}

    def save_reminders(self) -> None:
        """Сохраняет напоминания в файл хранения"""
        try:
            with open(self.reminders_file, 'w', encoding='utf-8') as f:
                json.dump(self.reminders, f, ensure_ascii=False)
        except Exception as e:
            logging.exception(f"Ошибка при сохранении напоминаний: {e}")

    async def check_reminders(self, helper: Any) -> None:
        """
        Проверка и отправка напоминаний
        """
        current_time = datetime.now()
        processed_reminders = False

        self.load_reminders()

        for user_id, user_reminders in list(self.reminders.items()):
            for reminder_id, reminder in list(user_reminders.items()):
                # Convert reminder time from ISO format string to datetime
                reminder_time = datetime.fromisoformat(reminder['time'])
                
                # Check if reminder time is in the past
                if current_time >= reminder_time:
                    await self.send_reminder(reminder, helper)
                    del self.reminders[user_id][reminder_id]
                    
                    # Remove user dict if no reminders left
                    if not self.reminders[user_id]:
                        del self.reminders[user_id]
        
                    # Save changes if any reminders were processed
                    processed_reminders = True

        if processed_reminders:
            self.save_reminders()

    async def send_reminder(self, reminder: Dict, helper) -> None:
        """
        Отправка напоминания через выбранную интеграцию
        """
        if reminder["integration"] == "telegram":
            await helper.send_message(
                chat_id=reminder["user_id"],
                text=f"🔔 Напоминание:\n{reminder['message']}",
                reply_to_message_id=reminder.get("reply_to_message_id")
            )
        # Здесь можно добавить другие интеграции (email, slack)

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """
        Выполнение функций плагина
        """
        user_id = str(kwargs.get('chat_id'))
        self.load_reminders()
        if function_name == "set_reminder":
            self.load_reminders()
            reminder_id = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{user_id}'
            # Convert datetime to ISO format string
            reminder_time = datetime.strptime(kwargs["time"], "%Y-%m-%d %H:%M")
            
            reminder = {
                "id": reminder_id,
                "user_id": user_id,
                "time": reminder_time.isoformat(),
                "message": kwargs["message"],
                "integration": kwargs["integration"],
                "reply_to_message_id": helper.message_id
            }

            if user_id not in self.reminders:
                self.reminders[user_id] = {}
            self.reminders[user_id][reminder_id] = reminder
            self.save_reminders()
            
            return {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": f"Напоминание установлено на {kwargs['time']}"
                }
            }

        elif function_name == "delete_reminder":
            self.load_reminders()
            reminder_id = kwargs.get("reminder_id") or kwargs.get("query")
            logging.info(f"Список напоминаний для пользователя {user_id}: {self.reminders}")
            logging.info(f"kwargs: {kwargs}")
            logging.info(f"reminder_id: {reminder_id}")

            if user_id in self.reminders and reminder_id in self.reminders[user_id]:
                del self.reminders[user_id][reminder_id]
                if not self.reminders[user_id]:
                    del self.reminders[user_id]
                self.save_reminders()
                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": f"Напоминание {reminder_id} удалено"
                    }
                }                
            
            return {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": "Напоминание не найдено"
                }
            }                

        return {"error": "Unknown function"}

    async def handle_reminder_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик callback-запросов от кнопок напоминаний"""
        query = update.callback_query

        try:
            action, command, reminder_id = query.data.split(":")
            
            # Обработка закрытия меню
            if action == "reminder" and command == "close_menu":
                await query.answer("Меню закрыто")
                await query.message.delete()
                return

            if action == "reminder" and command == "view":
                user_id = str(query.from_user.id)
                
                # Проверяем существование напоминания
                if user_id in self.reminders and reminder_id in self.reminders[user_id]:
                    reminder = self.reminders[user_id][reminder_id]
                    reminder_time = datetime.fromisoformat(reminder['time'])
                    formatted_time = reminder_time.strftime('%d.%m.%Y %H:%M')
                    
                    # Показываем детали напоминания во всплывающем окне
                    await query.answer(
                        text=f"📅 {formatted_time}\n📝 {reminder['message']}",
                        show_alert=True,
                        cache_time=0
                    )
                    return
                else:
                    await query.answer("Напоминание не найдено", show_alert=True)
                    return

            if action == "reminder" and command == "delete":
                user_id = str(query.from_user.id)
                await query.answer()  # Отвечаем на callback запрос для удаления
                
                # Проверяем существование напоминания
                if user_id in self.reminders and reminder_id in self.reminders[user_id]:
                    # Удаляем напоминание
                    del self.reminders[user_id][reminder_id]
                    if not self.reminders[user_id]:
                        del self.reminders[user_id]
                    self.save_reminders()
                    
                    # Обновляем сообщение со списком напоминаний
                    if user_id in self.reminders:
                        keyboard = []
                        
                        for r_id, r in self.reminders[user_id].items():      
                            reminder_time = datetime.fromisoformat(r['time'])
                            formatted_time = reminder_time.strftime('%d.%m.%Y %H:%M')
                            
                            # Создаем ряд из двух кнопок для каждого напоминания
                            keyboard.append([
                                InlineKeyboardButton(
                                    text=f"📅 {formatted_time} 📝 {r['message']}",
                                    callback_data=f"reminder:view:{r_id}"
                                ),
                                InlineKeyboardButton(
                                    text="❌ Удалить",
                                    callback_data=f"reminder:delete:{r_id}"
                                ),
                            ])
                        
                        # Добавляем кнопку закрытия
                        keyboard.append([
                            InlineKeyboardButton("❌ Закрыть", callback_data="reminder:close_menu:")
                        ])
                        await query.edit_message_text(
                            text="Ваши напоминания:",
                            reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
                            parse_mode='markdown'
                        )
                    else:
                        await query.edit_message_text(
                            text="У вас нет активных напоминаний.",
                            parse_mode='markdown'
                        )
                    
                    return
                    
        except Exception as e:
            logging.error(f"Ошибка при обработке callback запроса: {e}")
            await query.edit_message_text(
                text=f"Произошла ошибка при удалении напоминания: {str(e)}",
                parse_mode='markdown'
            )