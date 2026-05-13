import base64
import os
import logging
import aiohttp
import json
from typing import Dict
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound
from ..model_constants import LLMGATEWAY_HIGH_MODEL
from .plugin import Plugin

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

lexer = None

class GitHubCodeAnalysisPlugin(Plugin):

    def __init__(self):
        super().__init__()
        self.model = LLMGATEWAY_HIGH_MODEL
        self.max_tokens = int(os.environ.get('MAX_TOKENS', 1000))
        self.temperature = float(os.environ.get('TEMPERATURE', 1.0))

    def get_source_name(self) -> str:
        return 'GitHub Code Analysis'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'analyze_github_code',
                'description': (
                    'Analyze code from a GitHub repository: fetches the file (or repository '
                    'overview when path is empty) via the GitHub API and runs an LLM analysis '
                    'against the user\'s question. Use when the user wants insights about '
                    'code structure, quality, or behavior of a specific repo/file.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "User\'s analysis question or instruction, e.g. 'find potential "
                                "bugs', 'explain the auth flow', 'summarize what this file does'."
                            ),
                        },
                        'owner': {'type': 'string', 'description': 'Repository owner (GitHub username or organization).'},
                        'repo': {'type': 'string', 'description': 'Repository name.'},
                        'path': {
                            'type': 'string',
                            'description': 'Path to a specific file inside the repository. Leave empty to fetch the repo root listing.',
                            'default': '',
                        },
                    },
                    'required': ['prompt', 'owner', 'repo'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        if function_name == 'analyze_github_code':
            return await self.analyze_github_code(**kwargs)
        return {'error': 'Unknown function name'}

    async def analyze_github_code(self, owner, repo, path='', prompt=''):
        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
        headers = ""

    
        logging.info(f"Requesting URL: {url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                logging.info(f"GIHUB GIHUB GIHUB Response status: {response.status}")
            
                if response.status != 200:
                    logging.info(f"Failed to fetch repository contents: {response.status}")
                    return {'error': f'Failed to fetch repository contents: {response.status}'}
            
                contents = await response.read()  # Чтение сырых байтов



                try:
                    data = contents.decode('utf-8')
                    contents = json.loads(data)  # Явное преобразование в JSON
                except UnicodeDecodeError as e:
                    logging.error(f"Decoding error: {e}")
                    contents = ""
        
                if isinstance(contents, dict) and 'message' in contents:
                    logging.info(f"GitHub error: {contents['message']}")
                    return {'error': f'GitHub error: {contents["message"]}'}

                analysis_results = []

                if isinstance(contents, dict):  # Если ответ не список, преобразуем в список
                    contents = [contents]

                for item in contents:
                    if item['type'] == 'file':
                        logging.info(f"Analyzing file: {item['name']}")

                        if 'content' in item and item['encoding'] == 'base64':
                            try:
                                code = base64.b64decode(item['content']).decode('utf-8')
                            except (UnicodeDecodeError, base64.binascii.Error) as e:
                                logging.error(f"Error decoding Base64 content for {item['name']}: {e}")
                                continue

                            language = self.detect_language(item['name'], code)
                            logging.info(f"Detected language for {item['name']}: {language}")

                            if language:
                                analysis = await self.analyze_code_with_chatgpt(code, language, prompt)
                                analysis_results.append({
                                    'file': item['name'],
                                    'language': language,
                                    'analysis': analysis
                                })

                return {'results': analysis_results}
            
    def detect_language(self, filename, code):
        try:
            logging.info(f"GITHUB filename: {filename}")
            lexer = get_lexer_for_filename(filename, code)
            logging.info(f"Detected lexer: {lexer.name}")
            if lexer.name in ('C', 'Arduino', 'Objective-C', 'Python', 'HTML', 'CSS', 'JavaScript', 'TypeScript', 'PHP', 'Ruby', 'Java', 'Kotlin', 'Swift', 'Go', 'Rust', 'C#', 'Visual Basic', 'SQL', 'Markdown', 'YAML', 'JSON', 'XML', 'SVG', 'TeX', 'Assembly', 'Haskell', 'Erlang', 'Elixir', 'OCaml', 'Scala', 'Groovy', 'R', 'MATLAB', 'Perl', 'Tcl', 'Lua', 'Erlang', 'Elixir', 'OCaml', 'Scala', 'Groovy', 'R', 'MATLAB', 'Perl', 'Tcl', 'Lua'):
                 return lexer.name
            else:
                return None
        except ClassNotFound as e:
            logging.info(f"Lexer not found for the file {filename}: {str(e)}")
            return None
        except Exception as e:
            logging.info(f"An unexpected error occurred: {str(e)}")
            return None

    async def analyze_code_with_chatgpt(self, code, language, prompt):
        if prompt == "":
            prompt = "Provide insights and suggestions"
        logging.info(f"prompt = {prompt}")
        prompt = f"We have some code written in {language}. {prompt}:\n{code}\n"

        completion = await self.openai.chat_completion(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return completion.choices[0].message.content.strip()
