# Publicar no GitHub

Este pacote **ainda nao foi publicado**. A integracao disponivel na conversa oferece leitura, nao criacao de repositorios ou push. O script so publica quando executado com `--publish`.

## Preparacao

Requer Python 3.11+, Git e GitHub CLI (`gh`). Extraia esta pasta fora de outro repositorio Git. Autentique o GitHub CLI na conta `mayklink`, sem compartilhar tokens no chat:

```powershell
gh auth login --hostname github.com --git-protocol https --web
```

Abra o PowerShell na pasta que contem `publish_github.py` e execute:

```powershell
py .\publish_github.py --publish
```

O script valida o manifesto, confirma a conta, recusa sobrescrever um repositorio existente, cria `mayklink/ai-player-lab` como **privado**, prepara um commit na branch `main`, envia os arquivos e compara o SHA remoto com o local. Ele nao faz force-push, nao apaga repositorios, nao altera configuracao Git global, nao instala dependencias e nao chama Jev ou modelos multimodais.

Somente arquivos enumerados no manifesto sao preparados para commit. Os arquivos originais do prototipo, testes, documentacao e evidencias sinteticas foram preservados. `.env.example` contem somente nomes de variaveis sem chaves. Nenhum arquivo `.env`, `runs/`, `vendor/`, `models/` ou ambiente virtual sera incluido.

Para verificar apenas os arquivos, sem acessar GitHub:

```powershell
py .\publish_github.py
```

Em caso de falha de rede durante a criacao/envio, o repositorio pode ter sido criado sem receber o commit. O script preserva o trabalho local e nao sobrescreve nada automaticamente: inspecione o estado antes de repetir.

## Validacao desta entrega

A verificacao local de hashes e a compilacao do script foram executadas no ambiente de preparacao. **Criacao remota, autenticacao real e push nao foram executados.** Esta entrega nao acrescenta testes reais de gameplay ou de APIs ao prototipo.

Documentacao oficial: https://cli.github.com/manual/gh_repo_create e https://cli.github.com/manual/gh_auth_login.
