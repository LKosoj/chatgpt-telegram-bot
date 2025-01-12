from typing import Dict, Optional, List
import os
import requests
import logging
import random
from plugins.plugin import Plugin


class MovieInfoPlugin(Plugin):
    """
    Плагин для поиска и анализа информации о фильмах
    """

    def __init__(self):
        # Загрузка API ключей из переменных окружения
        self.TMDB_API_KEY = os.getenv('TMDB_API_KEY')
        
        if not self.TMDB_API_KEY:
            raise ValueError("Необходимо установить API ключи для TMDb")
        
        self.TMDB_BASE_URL = "https://api.themoviedb.org/3"
        
        # Словарь жанров TMDb
        self.GENRES = {
            28: "Боевик", 12: "Приключения", 16: "Анимация", 35: "Комедия", 
            80: "Криминал", 99: "Документальный", 18: "Драма", 10751: "Семейный", 
            14: "Фэнтези", 36: "История", 27: "Ужасы", 10402: "Музыка", 
            9648: "Мистика", 10749: "Романтика", 878: "Научная фантастика", 
            10770: "Телефильм", 53: "Триллер", 10752: "Военный", 37: "Вестерн"
        }

    def get_source_name(self) -> str:
        return 'Movie Information'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'name': 'get_new_movies',
                'description': 'Получает список новых фильмов, выходящих в прокат',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'genre': {
                            'type': 'string', 
                            'description': f'Жанр фильма. Доступные жанры: {", ".join(self.GENRES.values())}'
                        },
                        'count': {
                            'type': 'integer', 
                            'description': 'Количество фильмов для возврата (по умолчанию 20)'
                        }
                    },
                    'required': [],
                },
            },
            {
                'name': 'get_movie_recommendations',
                'description': 'Получает рекомендации по фильмам с использованием ИИ',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'genre': {
                            'type': 'string', 
                            'description': f'Жанр фильма. Доступные жанры: {", ".join(self.GENRES.values())}'
                        },
                        'count': {
                            'type': 'integer', 
                            'description': 'Количество фильмов для анализа (по умолчанию 20)'
                        }
                    },
                    'required': [],
                },
            }
        ]

    def _get_genre_id(self, genre_name: Optional[str] = None) -> Optional[int]:
        """
        Получает ID жанра по его названию
        """
        if not genre_name:
            return None
        
        # Приводим к нижнему регистру для нечувствительного поиска
        genre_name = genre_name.lower()
        
        # Ищем совпадение
        for genre_id, genre_title in self.GENRES.items():
            if genre_name == genre_title.lower():
                return genre_id
        
        return None

    def _get_new_movies(self, genre: Optional[str] = None, count: int = 20):
        """
        Внутренний метод для получения новых фильмов
        """
        url = f"{self.TMDB_BASE_URL}/movie/now_playing"
        params = {
            "api_key": self.TMDB_API_KEY, 
            "language": "ru-RU", 
            "region": "RU"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            movies = response.json().get("results", [])
            
            # Фильтрация по жанру, если указан
            if genre:
                genre_id = self._get_genre_id(genre)
                if genre_id:
                    movies = [
                        movie for movie in movies 
                        if any(genre_id == g for g in movie.get('genre_ids', []))
                    ]
            return movies[:count]
        
        except requests.RequestException as e:
            logging.error(f"Ошибка при запросе данных: {str(e)}")
            return []

    def _get_movie_details(self, movie_id):
        """
        Получает подробную информацию о фильме
        """
        url = f"{self.TMDB_BASE_URL}/movie/{movie_id}"
        params = {"api_key": self.TMDB_API_KEY, "language": "ru-RU"}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            details = response.json()
            
            # Форматируем дату выхода
            release_date = details.get('release_date', 'Дата не указана')
            try:
                from datetime import datetime
                parsed_date = datetime.strptime(release_date, '%Y-%m-%d')
                details['formatted_release_date'] = parsed_date.strftime('%d %B %Y')
            except (ValueError, TypeError):
                details['formatted_release_date'] = release_date
            
            return details
        except requests.RequestException as e:
            logging.error(f"Ошибка при получении данных о фильме {movie_id}: {str(e)}")
            return {}

    def _get_movie_reviews(self, movie_id):
        """
        Получает отзывы о фильме
        """
        url = f"{self.TMDB_BASE_URL}/movie/{movie_id}/reviews"
        params = {"api_key": self.TMDB_API_KEY, "language": "ru-RU"}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.RequestException as e:
            logging.error(f"Ошибка при получении отзывов для фильма {movie_id}: {str(e)}")
            return []

    def _discover_movies(self, genre: Optional[str] = None, count: int = 20):
        """
        Внутренний метод для расширенного поиска фильмов
        """
        url = f"{self.TMDB_BASE_URL}/discover/movie"
        sort_by_options = [
            'popularity.desc', 
            'vote_count.desc',
            'revenue.desc'
        ]
        sort = random.choice(sort_by_options)
        page = random.randint(1, 15)
        # Получаем ID жанра, если указан
        genre_id = self._get_genre_id(genre) if genre else None
        
        params = {
            "api_key": self.TMDB_API_KEY, 
            "language": "ru-RU", 
            "page": page,
            "sort_by": sort,
            "include_adult": "true",
            "include_video": "false"
        }
        
        # Добавляем фильтр по жанру, если указан
        if genre_id:
            params["with_genres"] = genre_id
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            movies = response.json().get("results", [])
            
            return movies[:count]
        
        except requests.RequestException as e:
            logging.error(f"Ошибка при запросе данных: {str(e)}")
            return []

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            if function_name == 'get_new_movies':
                # Извлекаем параметры с значениями по умолчанию
                genre = kwargs.get('genre')
                count = kwargs.get('count', 30)
                
                movies = self._get_new_movies(genre=genre, count=count)
                return {
                    'movies': movies,
                    'genre_filter': genre or 'Все жанры'
                }
                        
            elif function_name == 'get_movie_recommendations':
                # Извлекаем параметры с значениями по умолчанию
                genre = kwargs.get('genre')
                count = kwargs.get('count', 10)
                
                # Получаем новые фильмы с учетом жанра
                movies = self._get_new_movies(genre=genre, count=count)
                # Добавляем фильмы из расширенного поиска
                movies.extend(self._discover_movies(genre=genre, count=count))
                
                # Подготавливаем данные для анализа
                movie_data = []
                for movie in movies:
                    # Безопасное извлечение данных
                    movie_id = movie.get('id')
                    if not movie_id:
                        continue
                    
                    details = self._get_movie_details(movie_id) or {}
                    reviews = self._get_movie_reviews(movie_id)
                    
                    critic_reviews = "\n".join(
                        [f"- {review.get('author', 'Аноним')}: {review.get('content', 'Нет текста')[:200]}..." 
                         for review in reviews[:3]]
                    ) if reviews else "Нет отзывов критиков."
                    
                    # Безопасное извлечение жанров
                    genre_ids = movie.get('genre_ids', [])
                    genres = ", ".join([
                        self.GENRES.get(genre_id, str(genre_id)) 
                        for genre_id in genre_ids
                    ])
                    
                    movie_data.append({
                        "title": movie.get('title', 'Название неизвестно'),
                        "overview": details.get("overview", "Нет описания."),
                        "rating": details.get("vote_average", "Нет данных"),
                        "genres": genres,
                        "critic_reviews": critic_reviews,
                        "release_date": details.get('formatted_release_date', 'Дата не указана')
                    })
                
                # Формируем промпт для GPT
                movie_descriptions = "\n\n".join(
                    [f"Название: {movie['title']}\nДата выхода: {movie['release_date']}\nОписание: {movie['overview']}\nРейтинг: {movie['rating']}\nЖанры: {movie['genres']}\nОтзывы критиков:\n{movie['critic_reviews']}"
                     for movie in movie_data]
                )
                logging.info(f"Список фильмов для анализа: {movie_descriptions}")
                # Добавляем информацию о жанре в промпт
                genre_info = f" в жанре {genre}" if genre else ""
                
                prompt = f"""
                У меня есть список фильмов{genre_info}. Проанализируй их описания, даты выхода, рейтинги, жанры и отзывы критиков и порекомендуй мне лучшие фильмы для просмотра. 
                Вот список фильмов:

                {movie_descriptions}

                Пожалуйста, предложи свои рекомендации и объясни, почему ты выбрал именно эти фильмы.
                """
                
                try:
                    chat_id = kwargs.get('chat_id')
                    response, tokens = await helper.ask(prompt, chat_id)
                    
                    return {
                        'recommendations': response,
                        'analyzed_movies': movie_data,
                        'genre_filter': genre or 'Все жанры',
                        'tokens_used': tokens
                    }
                except Exception as e:
                    logging.error(f"Ошибка при работе с OpenAI: {str(e)}")
                    return {'error': f'Ошибка при работе с OpenAI: {str(e)}'}
            
            else:
                return {'error': 'Неизвестный метод'}
        
        except Exception as e:
            logging.error(f"Непредвиденная ошибка: {str(e)}")
            return {'error': f'Непредвиденная ошибка: {str(e)}'} 