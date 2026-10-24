#!/usr/bin/env python3
"""Publish this verified prototype to a NEW private GitHub repository.

Default: validate package offline. Only --publish accesses GitHub.
Requires Python 3.11+, Git and an authenticated GitHub CLI (gh).
Does not install dependencies, run the game, call model APIs or read API keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OWNER = 'mayklink'
REPO = 'ai-player-lab'
FULL_NAME = f'{OWNER}/{REPO}'


class PublishError(RuntimeError):
    pass


def validate_package() -> list[str]:
    manifest = ROOT / 'MANIFEST.sha256'
    if not manifest.is_file():
        raise PublishError('MANIFEST.sha256 nao encontrado. Extraia o pacote completo.')
    paths = []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            expected, rel = line.split('  ', 1)
        except ValueError as exc:
            raise PublishError('Manifesto invalido.') from exc
        part = PurePosixPath(rel)
        if part.is_absolute() or '..' in part.parts or '\\' in rel:
            raise PublishError('Caminho inseguro no manifesto.')
        if part.name.startswith('.env') and part.name != '.env.example':
            raise PublishError('Um arquivo de segredos nao pode ser publicado.')
        if any(p in {'.git', '.venv', 'runs', 'models', 'vendor', '__pycache__'} for p in part.parts):
            raise PublishError(f'Diretorio de execucao nao permitido: {rel}')
        file = ROOT.joinpath(*part.parts)
        if file.is_symlink() or not file.resolve().is_relative_to(ROOT):
            raise PublishError(f'Caminho externo nao permitido: {rel}')
        if not file.is_file():
            raise PublishError(f'Arquivo ausente: {rel}')
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != expected:
            raise PublishError(f'Arquivo alterado: {rel}. Use uma extracao nova do pacote.')
        paths.append(rel)
    if len(set(paths)) != len(paths):
        raise PublishError('Entradas duplicadas no manifesto.')
    return paths + ['MANIFEST.sha256']


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Commands must target the same host checked below, not an Enterprise host.
    env['GH_HOST'] = 'github.com'
    env['GH_PROMPT_DISABLED'] = '1'
    try:
        proc = subprocess.run(
            list(args), cwd=ROOT, env=env, text=True,
            encoding='utf-8', errors='replace', capture_output=True,
            timeout=180, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PublishError(
            'Tempo esgotado. A operacao remota pode ter ocorrido parcialmente. '
            'Confira o GitHub antes de repetir; nada sera apagado automaticamente.'
        ) from exc
    if check and proc.returncode:
        detail = (proc.stderr or proc.stdout).strip()
        raise PublishError(f'Falha em {args[0]} {args[1]}:\n{detail}')
    return proc


def publish(paths: list[str]) -> None:
    for exe in ('git', 'gh'):
        if not shutil.which(exe):
            raise PublishError(f'{exe} nao encontrado. Instale Git e GitHub CLI e tente novamente.')
    auth = run('gh', 'auth', 'status', '--hostname', 'github.com', check=False)
    if auth.returncode:
        raise PublishError(
            'GitHub CLI nao autenticado. Execute:\n'
            'gh auth login --hostname github.com --git-protocol https --web\n'
            'Depois rode este script novamente. Nao cole tokens no chat.'
        )
    profile = json.loads(run('gh', 'api', '--hostname', 'github.com', 'user').stdout)
    if profile.get('login') != OWNER:
        raise PublishError(
            f'A conta ativa nao e {OWNER}. Execute gh auth switch --hostname github.com '
            f'--user {OWNER}, ou autentique essa conta, e tente novamente.'
        )
    existing = run('gh', 'api', '--hostname', 'github.com', f'repos/{FULL_NAME}', check=False)
    if existing.returncode == 0:
        raise PublishError(
            f'{FULL_NAME} ja existe. O script parou sem alterar esse repositorio.'
        )
    if '404' not in (existing.stderr + existing.stdout):
        raise PublishError(
            'Nao foi possivel confirmar que o nome do repositorio esta disponivel. '
            'Confira a conexao e as permissoes. Nenhuma publicacao foi iniciada.'
        )
    if (ROOT / '.git').exists():
        raise PublishError(
            'Esta pasta ja tem .git. Use uma extracao nova ou inspecione a tentativa anterior; '
            'o script nao sobrescreve historico ou remotes existentes.'
        )
    # Refuse to initialize inside a different existing repository.
    enclosing = run('git', 'rev-parse', '--show-toplevel', check=False)
    if enclosing.returncode == 0:
        raise PublishError('Extraia o pacote fora de qualquer repositorio Git existente.')
    print(f'Criando repositorio local para {FULL_NAME} (privado)...')
    run('git', 'init', '-b', 'main')
    # Commit identity is local to this new repo. No private e-mail is read.
    run('git', 'config', '--local', 'user.name', profile.get('name') or OWNER)
    run('git', 'config', '--local', 'user.email', f"{profile['id']}+{OWNER}@users.noreply.github.com")
    run('git', 'config', '--local', 'core.autocrlf', 'false')
    # Only verified package paths are staged, never unlisted local files.
    run('git', 'add', '--', *paths)
    staged = run('git', 'diff', '--cached', '--name-only', '-z').stdout.split('\0')
    staged = {p for p in staged if p}
    if staged != set(paths):
        raise PublishError('O conjunto de arquivos preparados difere do manifesto. Publicacao cancelada.')
    run('git', 'commit', '-m', 'Initial prototype v0.1.1: local vision, Jev and multimodal APIs')
    print('Criando repositorio privado e enviando commit...')
    try:
        run(
            'gh', 'repo', 'create', FULL_NAME, '--private', '--source=.',
            '--remote=origin', '--push', '--description',
            'Experimental visual game agent: local perception, Jev API and asynchronous multimodal API memory.',
        )
    except PublishError as exc:
        raise PublishError(
            f'{exc}\nO repositorio remoto pode ter sido criado sem o envio completo. '
            f'Confira {FULL_NAME}. O commit local foi preservado; nao use force-push.'
        ) from exc
    metadata = json.loads(run('gh', 'api', '--hostname', 'github.com', f'repos/{FULL_NAME}').stdout)
    if metadata.get('private') is not True:
        raise PublishError('Nao foi possivel confirmar a visibilidade privada. Confira o GitHub.')
    local_sha = run('git', 'rev-parse', 'HEAD').stdout.strip()
    remote_sha = run('gh', 'api', '--hostname', 'github.com', f'repos/{FULL_NAME}/commits/main', '--jq', '.sha').stdout.strip()
    if local_sha != remote_sha:
        raise PublishError('O commit remoto nao corresponde ao local. Confira o envio.')
    print(f"Publicado e verificado: {metadata['html_url']}")
    print(f'Branch: main | Commit: {local_sha} | Arquivos: {len(paths)} | Privado: sim')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', help='Cria repositorio privado e faz push real.')
    args = parser.parse_args()
    try:
        paths = validate_package()
        print(f'Pacote validado: {len(paths)} arquivos. Sem chamada a APIs de modelos.')
        if not args.publish:
            print('Somente verificacao local. Para publicar: py publish_github.py --publish')
            return 0
        publish(paths)
        return 0
    except (PublishError, OSError, ValueError) as exc:
        print(f'ERRO: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
