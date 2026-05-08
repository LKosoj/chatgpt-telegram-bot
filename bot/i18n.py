import json
import logging
import os
from contextvars import ContextVar
from functools import lru_cache

logger = logging.getLogger(__name__)
DEFAULT_LANGUAGE = 'en'
AUTO_LANGUAGE = 'auto'
_current_language = ContextVar('current_language', default=None)
LANGUAGE_NAMES = {
    'en': 'English',
    'ar': 'العربية',
    'de': 'Deutsch',
    'es': 'Español',
    'fa': 'فارسی',
    'fi': 'Suomi',
    'he': 'עברית',
    'id': 'Bahasa Indonesia',
    'it': 'Italiano',
    'ms': 'Bahasa Melayu',
    'nl': 'Nederlands',
    'pl': 'Polski',
    'pt-br': 'Português (BR)',
    'ru': 'Русский',
    'tr': 'Türkçe',
    'uk': 'Українська',
    'uz': 'O‘zbek',
    'vi': 'Tiếng Việt',
    'zh-cn': '简体中文',
    'zh-tw': '繁體中文',
}
LANGUAGE_ALIASES = {
    'iw': 'he',
    'in': 'id',
    'pt': 'pt-br',
    'zh-hans': 'zh-cn',
    'zh-sg': 'zh-cn',
    'zh-hant': 'zh-tw',
    'zh-hk': 'zh-tw',
    'zh-mo': 'zh-tw',
}

# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def supported_languages():
    return tuple(translations.keys())


def language_name(language):
    language = normalize_language(language)
    return LANGUAGE_NAMES.get(language, language)


def is_auto_language(bot_language):
    return str(bot_language or AUTO_LANGUAGE).strip().lower() == AUTO_LANGUAGE


def configured_language(bot_language):
    if is_auto_language(bot_language):
        return AUTO_LANGUAGE
    return normalize_language(bot_language)


def set_current_language(bot_language):
    return _current_language.set(normalize_language(bot_language))


def reset_current_language(token):
    _current_language.reset(token)


def get_current_language():
    return _current_language.get() or DEFAULT_LANGUAGE


@lru_cache(maxsize=None)
def normalize_language(bot_language):
    language = str(bot_language or DEFAULT_LANGUAGE).strip().lower().replace('_', '-')
    language = LANGUAGE_ALIASES.get(language, language)
    if language in translations:
        return language

    base_language = language.split('-', 1)[0]
    base_language = LANGUAGE_ALIASES.get(base_language, base_language)
    if base_language in translations:
        return base_language

    logger.warning(
        "Unsupported bot_language code '%s'. Falling back to '%s'. Supported languages: %s",
        bot_language,
        DEFAULT_LANGUAGE,
        ', '.join(supported_languages()),
    )
    return DEFAULT_LANGUAGE


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    current_language = _current_language.get()
    if current_language:
        language = current_language
    elif is_auto_language(bot_language):
        language = DEFAULT_LANGUAGE
    else:
        language = normalize_language(bot_language)
    try:
        return translations[language][key]
    except KeyError:
        logger.warning(f"No translation available for bot_language code '{language}' and key '{key}'")
        # Fallback to English if the translation is not available
        if key in translations[DEFAULT_LANGUAGE]:
            return translations[DEFAULT_LANGUAGE][key]
        else:
            logger.warning(f"No english definition found for key '{key}' in translations.json")
            # return key as text
            return key
