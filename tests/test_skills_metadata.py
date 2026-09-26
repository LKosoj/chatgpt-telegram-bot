"""Проверки метаданных реальных skills под bot/skills/** (T06, план §8.3).

В отличие от tests/test_skills_plugin.py (который использует синтетические SKILL.md во
временных каталогах), эти тесты сканируют настоящий bot/skills/** через тот же
SkillsPlugin._scan_skills() / _parse_skill_markdown(), что и в проде — чтобы поймать
регрессии реорганизации META-SKILLS (T06 §7) на реальных файлах.
"""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

if importlib.util.find_spec("markdown2") is None:
    _markdown2 = types.ModuleType("markdown2")
    _markdown2.__spec__ = importlib.machinery.ModuleSpec("markdown2", loader=None)
    _markdown2.markdown = lambda text, *args, **kwargs: text
    sys.modules["markdown2"] = _markdown2

from bot.plugins.skills import SkillsPlugin

REAL_SKILLS_DIR = Path(__file__).resolve().parents[1] / "bot" / "skills"


@pytest.fixture()
def real_skills(tmp_path, monkeypatch):
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    monkeypatch.setenv("SKILLS_DIR", str(REAL_SKILLS_DIR))
    plugin = SkillsPlugin()
    plugin.initialize(storage_root=str(storage_dir))
    return plugin.available_skills


def test_real_skills_have_non_empty_name_and_description(real_skills):
    assert real_skills, "expected at least one skill under bot/skills/**"
    for skill_id, skill in real_skills.items():
        assert skill["name"].strip(), f"{skill_id}: empty name"
        assert skill["description"].strip(), f"{skill_id}: empty description"


def test_real_skill_ids_are_unique(real_skills):
    # available_skills — уже dict, ключи по построению уникальны, поэтому сравнивать их
    # напрямую бессмысленно (см. T06-review.md, раунд 1, NIT 2). Реальная регрессия, которую
    # стоит ловить, — два разных skill_id с одинаковым frontmatter `name:` (например, если
    # после дедупа META-SKILLS в T06 §7.2 где-то останется забытый дубль каталога с новым id,
    # но старым name) — модель видит только name, а не id, так что такая коллизия для неё
    # неотличима от одного и того же skill.
    names = [info["name"] for info in real_skills.values()]
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, f"duplicate skill name(s) across different skill ids: {duplicates}"


def test_decision_framework_description_is_within_240_chars(real_skills):
    assert "decision-framework" in real_skills
    assert len(real_skills["decision-framework"]["description"]) <= 240
