# AI Player Lab

Laboratório experimental em Python para estudar agentes que observam jogos 2D
pelos pixels, mantêm estado temporal e propõem ações limitadas. A percepção
local usa OpenCV; as decisões podem usar Jev por API e um modelo multimodal
remoto para navegação e revisão de episódios.

O projeto registra experiências para consultar em tentativas posteriores.
Isso é memória de episódios e hipóteses, não treinamento dos pesos dos modelos.
Não lê a memória interna do emulador.

**Estado atual:** protótipo de pesquisa, com foco experimental em Super Mario
World. Não há garantia de jogar sozinho, vencer fases ou melhorar a cada
partida. Jogos, ROMs, imagens de referência e credenciais não estão incluídos.
Sem imagens locais rotuladas, o perfil de Mario tem reconhecimento limitado e
pode permanecer bloqueado. O caminho reproduzível de entrada é o teste
sintético abaixo, que não exige jogo nem APIs.

## Requisitos

- Python **3.11 ou superior** e Git.
- Windows é a plataforma de referência para os comandos abaixo. O controle de
  teclado do desktop é específico de Windows; outras plataformas não têm
  validação equivalente de ponta a ponta.
- Internet para instalar dependências e, se habilitadas, acessar as APIs.
- Chromium instalado pelo Playwright para o modo de navegador.
- Nenhuma GPU dedicada ou modelo local é exigido pelo teste sintético.

Execute os comandos na raiz do repositório. Não é necessário ativar o ambiente
virtual; os exemplos usam seu executável diretamente.

## Instalar e executar o primeiro teste

No PowerShell:

```powershell
git clone https://github.com/mayklink/ai-player-lab.git
cd ai-player-lab
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ai_player_lab doctor
.\.venv\Scripts\python.exe -m ai_player_lab benchmark --frames 180
.\.venv\Scripts\python.exe -m pytest -q
```

Se você usa outra versão compatível, substitua `py -3.11` pelo executável dela.
O comando `doctor` grava o diagnóstico em `runs/machine.json`. O benchmark
processa uma cena gerada por código e grava métricas, observações e vídeo em
`runs/synthetic/`. Esses comandos não chamam APIs nem controlam o teclado.

Nos testes, os módulos que exigem imagens privadas são indicados como
ignorados quando elas estão ausentes. Isso é esperado em um clone público;
passar nos testes restantes não comprova gameplay real.

## Configurar as APIs

Esta etapa só é necessária para os modos que usam modelos remotos.

```powershell
Copy-Item .env.example .env
```

Edite `.env` localmente:

```dotenv
TYPESAFE_API_KEY=SUA_CHAVE_JEV
VLM_BASE_URL=https://seu-provedor.example/v1
VLM_MODEL=ID_DO_MODELO_COM_VISAO
VLM_API_KEY=SUA_CHAVE_MULTIMODAL
```

| Variável | Uso |
| --- | --- |
| `TYPESAFE_API_KEY` | Autenticação das decisões Jev. |
| `VLM_BASE_URL` | Base HTTPS da API multimodal, antes de `/chat/completions`. Substitua o endereço ilustrativo. |
| `VLM_MODEL` | Identificador de um modelo com suporte a imagens e ao contrato do cliente. |
| `VLM_API_KEY` | Chave do provedor multimodal. |
| `OPENAI_API_KEY` | Alternativa à `VLM_API_KEY` quando ela não estiver preenchida. Não seleciona endpoint nem modelo. |

O CLI lê `.env` no diretório de execução sem sobrescrever variáveis já
existentes na sessão. O cliente multimodal envia imagens em data URLs no
formato Chat Completions e solicita JSON; a compatibilidade com qualquer
provedor ou modelo não é garantida. Não há fallback para inferência local.

Chamadas reais podem gerar cobrança. Os tetos de chamadas são limites por
cliente/processo, não um orçamento financeiro global; reiniciar processos
renova os limites. Omitir `--enable-input` impede a execução de teclas, mas
**não desativa as APIs** em `browser-play` ou `play`.

## Abrir o painel no navegador

Instale a dependência opcional e o navegador do projeto:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[browser]"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m ai_player_lab browser-play --url "https://ENDERECO-DO-SEU-JOGO" --seconds 120
```

Substitua a URL por um jogo que você tenha autorização para usar e automatizar.
Esse comando abre uma sessão exclusiva de Chromium e o servidor do painel em
[http://127.0.0.1:8765/](http://127.0.0.1:8765/). A sessão começa pausada;
abra o painel, confira endereço, objetivo e controles e inicie a rodada.
O jogo precisa expor uma área de canvas capturável. Páginas e jogos diferentes
podem exigir adaptações; informar outra URL não cria um perfil visual completo.

O comando acima não habilita teclas automáticas. Para permitir a execução de
ações na página, depois de conferir a configuração:

```powershell
.\.venv\Scripts\python.exe -m ai_player_lab browser-play --url "https://ENDERECO-DO-SEU-JOGO" --enable-input --seconds 120
```

No Windows, `iniciar-jogo.cmd` é um atalho para `browser-play` **com entrada
habilitada** e a URL padrão definida no CLI. Ele não instala dependências.
Use o comando explícito acima para escolher sua URL e o modo de entrada.
O painel permite pausar e encerrar; o prazo limita a rodada após iniciá-la.
Use `--port 8766` se a porta padrão estiver ocupada.

### Perfil visual e limitações

As imagens de referência de Mario foram retiradas da distribuição. Quem
possuir os direitos necessários deve fornecer exemplos locais conforme
[docs/ASSETS.md](docs/ASSETS.md). O repositório não oferece um instalador de
imagens nem um pacote público de templates; essa preparação é uma etapa
manual e técnica. Configurar chaves de API não substitui os exemplos visuais.

Com um perfil adequado, a visão local reconhece cenas conhecidas, Jev propõe
ações durante o gameplay e o componente chamado Luna no código usa o modelo
multimodal configurado para telas desconhecidas e revisão de episódios.
Esse nome não instala nem escolhe um modelo por você.

O reconhecimento pode perder o personagem, confundir objetos ou não reconhecer
uma transição. Nesses casos o controle pode pausar. A meta de captura de 30 FPS
não é uma garantia; latência de rede, captura e processamento afetam o resultado.
Retornar ao mapa não prova morte, e a opinião do modelo não comprova conclusão
do objetivo. Não há validação de conclusão autônoma das fases nesta versão.

## Captura de desktop e replay

O fluxo de desktop é avançado: precisa de calibração, jogo visível e, para
`observe`, da integração externa Pixel-to-Action. Antes de obtê-la, leia
[NOTICE.md](NOTICE.md) e [a auditoria do upstream](docs/AUDITORIA_UPSTREAM.md);
o código externo não é distribuído nem coberto pela licença deste projeto.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
.\.venv\Scripts\python.exe scripts/bootstrap_upstream.py
.\.venv\Scripts\python.exe -m ai_player_lab calibrate
.\.venv\Scripts\python.exe -m ai_player_lab observe --seconds 60
```

A calibração pede a região do jogo e o personagem e grava `configs/local.json`.
O modo `observe` acima não chama modelos nem envia teclas. Acrescente `--jev`
e/ou `--vlm` para habilitar as APIs correspondentes. Mantenha o recorte livre
de outras janelas; mudanças de tamanho exigem recalibração.

O comando `play` usa navegação multimodal e decisões Jev. Para entrada real no
desktop, confira `window_title` e o mapeamento de teclas em `configs/local.json`
antes de usar `--enable-input`. F10 desarma e Esc encerra; atalhos globais
dependem de `pynput`. Esse fluxo pode afetar o teclado do desktop.

Para analisar um vídeo com calibração compatível:

```powershell
.\.venv\Scripts\python.exe -m ai_player_lab replay "caminho\partida.avi" --config configs/local.json --headless
```

Replay não envia teclas. Sem `--jev` e `--vlm`, também não chama modelos.
Consulte as opções disponíveis com `python -m ai_player_lab --help` ou
`python -m ai_player_lab browser-play --help`, usando o Python do ambiente.

## Dados e privacidade

`runs/` guarda logs, imagens, vídeos e memórias locais; `data/game_profiles/`
guarda perfis por jogo e `configs/local.json` guarda a calibração. Esses
caminhos e `.env` são ignorados pelo Git. As saídas podem conter dados privados:
não as publique automaticamente.

Quando habilitado, Jev recebe estado estruturado; o modelo multimodal recebe
imagens e contexto do jogo. Confira o tratamento desses dados pelo provedor.
A memória de experiências não é prova causal de que uma ação funcionou.

## Problemas comuns

| Sintoma | Verificação |
| --- | --- |
| `No module named ai_player_lab` | Execute na raiz e instale com o mesmo Python: `.\.venv\Scripts\python.exe -m pip install -e .`. |
| Chromium não encontrado | Execute `python -m playwright install chromium` com o Python do ambiente. |
| Erro de configuração da API | Confira `.env`, endpoint HTTPS, modelo e chave; variáveis da sessão têm prioridade. |
| Painel abre, mas o agente não avança | Confira o início da rodada, o motivo de pausa, os templates locais, a captura e o opt-in de entrada. |
| Cena ou personagem não reconhecidos | O clone público não contém imagens de referência. Veja `docs/ASSETS.md`; APIs não resolvem essa ausência automaticamente. |

## Documentação e contribuição

- [Arquitetura](docs/ARQUITETURA.md) e [requisitos](docs/REQUISITOS.md).
- [Perfis por jogo](docs/PERFIS_POR_JOGO.md) e [assets locais](docs/ASSETS.md).
- [Validação histórica e limites](docs/VALIDACAO.md): resultados antigos não
  equivalem à validação do clone público atual; os artefatos citados não são distribuídos.
- [Como contribuir](CONTRIBUTING.md).

Código de autoria do projeto sob [MIT](LICENSE). Dependências e materiais
externos mantêm seus próprios direitos; veja [NOTICE.md](NOTICE.md).
