"""Local configuration; secret values are never logged or copied to session reports."""
from pathlib import Path
import os
import re


def load_local_environment(path: Path = Path('.env')) -> None:
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            match = re.fullmatch(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*', line)
            if match:
                name, value = match.groups()
                if value[:1] in ('"', "'") and value[-1:] == value[:1]:
                    value = value[1:-1]
                os.environ.setdefault(name, value)
    if not os.environ.get('VLM_API_KEY') and os.environ.get('OPENAI_API_KEY'):
        os.environ['VLM_API_KEY'] = os.environ['OPENAI_API_KEY']
