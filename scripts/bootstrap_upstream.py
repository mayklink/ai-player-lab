"""Clone fixado do projeto original. Não altera nem relicencia o upstream."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'vendor'/'pixel-to-action'
URL = 'https://github.com/danjfarrell/pixel-to-action.git'
COMMIT = 'ca697eb9e681a6db657ecc395bb38731960e01f8'


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()


def main() -> int:
    try:
        if DEST.exists():
            current = run('git','rev-parse','HEAD',cwd=DEST)
            if current != COMMIT:
                raise RuntimeError(f'Base existente está em {current}; nada foi sobrescrito. Commit esperado: {COMMIT}')
            if run('git','status','--porcelain',cwd=DEST):
                raise RuntimeError('Base contém alterações locais; não será modificada automaticamente')
        else:
            DEST.parent.mkdir(parents=True,exist_ok=True)
            run('git','clone','--no-checkout',URL,str(DEST))
            run('git','checkout','--detach',COMMIT,cwd=DEST)
        print(f'Base original pronta: {DEST}\nCommit: {COMMIT}')
        print('A extensão fica separada em src/ai_player_lab; nenhuma licença foi atribuída ao código original.')
        return 0
    except (OSError,subprocess.CalledProcessError,RuntimeError) as exc:
        print(f'Não foi possível preparar o upstream: {exc}',file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
