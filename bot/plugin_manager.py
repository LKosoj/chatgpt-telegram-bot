import json
import logging

from plugins.reaction import ReactionPlugin
from plugins.website_content import WebsiteContentPlugin
from plugins.youtube_transcript import YoutubeTranscriptPlugin
from plugins.gtts_text_to_speech import GTTSTextToSpeech
from plugins.auto_tts import AutoTextToSpeech
from plugins.dice import DicePlugin
from plugins.youtube_audio_extractor import YouTubeAudioExtractorPlugin
from plugins.ddg_image_search import DDGImageSearchPlugin
from plugins.ddg_translate import DDGTranslatePlugin
from plugins.spotify import SpotifyPlugin
from plugins.crypto import CryptoPlugin
from plugins.weather import WeatherPlugin
from plugins.ddg_web_search import DDGWebSearchPlugin
from plugins.wolfram_alpha import WolframAlphaPlugin
from plugins.deepl import DeeplTranslatePlugin
from plugins.worldtimeapi import WorldTimeApiPlugin
from plugins.whois_ import WhoisPlugin
from plugins.webshot import WebshotPlugin
from plugins.iplocation import IpLocationPlugin
from plugins.github_analysis import GitHubCodeAnalysisPlugin
from plugins.stable_diffusion import StableDiffusionPlugin
from plugins.prompt_perfect import PromptPerfectPlugin
from plugins.show_me_diagrams import ShowMeDiagramsPlugin
from plugins.reminders import RemindersPlugin
from plugins.language_learning import LanguageLearningPlugin
from plugins.task_management import TaskManagementPlugin

GOOGLE = ("google/gemini-flash-1.5-8b",)

class PluginManager:
    """
    A class to manage the plugins and call the correct functions
    """

    def __init__(self, config):
        enabled_plugins = config.get('plugins', [])
        plugin_mapping = {
            'show_me_diagrams': ShowMeDiagramsPlugin,
            'reaction': ReactionPlugin,
            'worldtimeapi': WorldTimeApiPlugin,
            'youtube_transcript': YoutubeTranscriptPlugin,
            'wolfram': WolframAlphaPlugin,
            'weather': WeatherPlugin,
            'crypto': CryptoPlugin,
            'ddg_web_search': DDGWebSearchPlugin,
            'ddg_translate': DDGTranslatePlugin,
            'ddg_image_search': DDGImageSearchPlugin,
            'spotify': SpotifyPlugin,
            'youtube_audio_extractor': YouTubeAudioExtractorPlugin,
            'dice': DicePlugin,
            'deepl_translate': DeeplTranslatePlugin,
            'gtts_text_to_speech': GTTSTextToSpeech,
            'auto_tts': AutoTextToSpeech,
            'whois': WhoisPlugin,
            'webshot': WebshotPlugin,
            'iplocation': IpLocationPlugin,
            'github_analysis': GitHubCodeAnalysisPlugin,
            'prompt_perfect': PromptPerfectPlugin,
            'stable_diffusion': StableDiffusionPlugin,
            'website_content': WebsiteContentPlugin,
            'reminders': RemindersPlugin,
            'language_learning': LanguageLearningPlugin,
            'task_management': TaskManagementPlugin,
        }
        self.plugins = [plugin_mapping[plugin]() for plugin in enabled_plugins if plugin in plugin_mapping]

    def get_functions_specs(self, helper, model_to_use):
        """
        Return the list of function specs that can be called by the model
        """
        seen_functions = set()
        all_specs = []
        for plugin in self.plugins:
            specs = plugin.get_spec()
            #logging.info(f"Plugin {plugin.__class__.__name__} specs: {specs}")
            for spec in specs:
                if spec and spec.get('name') not in seen_functions:
                    seen_functions.add(spec.get('name'))
                    all_specs.append(spec)

        if model_to_use in (GOOGLE):
            return {"function_declarations": all_specs}
        return [{"type": "function", "function": spec} for spec in all_specs]

    async def call_function(self, function_name, helper, arguments):
        """
        Call a function based on the name and parameters provided
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return json.dumps({'error': f'Function {function_name} not found'})

        return json.dumps(await plugin.execute(function_name, helper, **json.loads(arguments)), default=str)

    def get_plugin_source_name(self, function_name) -> str:
        """
        Return the source name of the plugin
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return ''
        return plugin.get_source_name()

    def __get_plugin_by_function_name(self, function_name):
        return next((plugin for plugin in self.plugins
                    if function_name in map(lambda spec: spec.get('name'), plugin.get_spec())), None)

    def get_plugin(self, plugin_name):
        """
        Returns the plugin with the given source name
        
        :param plugin_name: The source name of the plugin
        :return: The plugin instance or None if not found
        """
        for plugin in self.plugins:
            if plugin.get_source_name().lower() == plugin_name.lower():
                return plugin
        return None