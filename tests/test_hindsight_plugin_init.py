from __future__ import annotations

from types import SimpleNamespace

from bot.plugins.hindsight_memory import HindsightClient, HindsightMemoryPlugin, LESSON_TYPE_VERIFIED


def test_initialize_sets_defaults_for_all_13_keys():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={})

    assert plugin.config['hindsight_base_url'] == ''
    assert plugin.config['hindsight_api_token'] == ''
    assert plugin.config['hindsight_enabled'] is False
    assert plugin.config['hindsight_auto_recall'] is True
    assert plugin.config['hindsight_auto_save'] is True
    assert plugin.config['hindsight_namespace'] == 'default'
    assert plugin.config['hindsight_bank_prefix'] == 'telegram-'
    assert plugin.config['hindsight_recall_budget'] == 'mid'
    assert plugin.config['hindsight_recall_max_tokens'] == 4096
    assert plugin.config['hindsight_memory_types'] == f'world,experience,{LESSON_TYPE_VERIFIED}'
    assert plugin.config['hindsight_async_store'] is True
    assert plugin.config['hindsight_timeout'] == 30.0
    assert plugin.config['hindsight_max_auto_save_items'] == 5


def test_initialize_keeps_overrides():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={
        'hindsight_recall_budget': 'high',
        'hindsight_timeout': 5.0,
    })

    assert plugin.config['hindsight_recall_budget'] == 'high'
    assert plugin.config['hindsight_timeout'] == 5.0
    # Other defaults still set
    assert plugin.config['hindsight_bank_prefix'] == 'telegram-'
    assert plugin.config['hindsight_recall_max_tokens'] == 4096


def test_initialize_constructs_client_iff_enabled():
    # Case (a): empty creds -> no client, not active
    plugin_disabled = HindsightMemoryPlugin()
    plugin_disabled.initialize(plugin_config={
        'hindsight_base_url': '',
        'hindsight_api_token': '',
    })
    assert plugin_disabled.client is None
    assert plugin_disabled.is_active is False

    # Case (b): both creds -> client present, active
    plugin_enabled = HindsightMemoryPlugin()
    plugin_enabled.initialize(plugin_config={
        'hindsight_base_url': 'http://x',
        'hindsight_api_token': 't',
    })
    assert isinstance(plugin_enabled.client, HindsightClient)
    assert plugin_enabled.is_active is True


def test_initialize_mirrors_to_openai_config():
    openai = SimpleNamespace(config={})
    plugin = HindsightMemoryPlugin()
    plugin.initialize(openai=openai, plugin_config={'hindsight_recall_budget': 'high'})

    assert openai.config['hindsight_recall_budget'] == 'high'
    assert 'hindsight_enabled' in openai.config
    assert openai.config['hindsight_bank_prefix'] == 'telegram-'


def test_bank_id_for_uses_prefix():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={'hindsight_bank_prefix': 'tg:'})
    assert plugin.bank_id_for(42) == 'tg:42'

    plugin_default = HindsightMemoryPlugin()
    plugin_default.initialize(plugin_config={})
    assert plugin_default.bank_id_for(42) == 'telegram-42'


def test_memory_types_parses_string_list_and_fallback():
    plugin = HindsightMemoryPlugin()
    plugin.initialize(plugin_config={'hindsight_memory_types': 'a,b'})
    assert plugin.memory_types == ['a', 'b']

    plugin_list = HindsightMemoryPlugin()
    plugin_list.initialize(plugin_config={'hindsight_memory_types': [' x ', 'y']})
    assert plugin_list.memory_types == ['x', 'y']

    plugin_fallback = HindsightMemoryPlugin()
    plugin_fallback.initialize(plugin_config={'hindsight_memory_types': 123})
    assert plugin_fallback.memory_types == ['world', 'experience', LESSON_TYPE_VERIFIED]

    plugin_candidates_only = HindsightMemoryPlugin()
    plugin_candidates_only.initialize(plugin_config={'hindsight_memory_types': 'lesson_candidate'})
    assert plugin_candidates_only.memory_types == ['world', 'experience', LESSON_TYPE_VERIFIED]
