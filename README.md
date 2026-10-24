# AI Player Lab · protótipo 0.1.1 · APIs remotas

**Pixels → estado temporal → Jev → execução limitada**, com **Jev e modelo multimodal por API**, revisão assíncrona e memória local em paralelo.

Preparado para Mayk em 24/10/2026. Primeiro alvo de avaliação: Super Mario World em navegador/emulador, somente pela imagem. **Ainda não foi validado em uma partida real.** Não inclui ROM, jogo, pesos de modelos ou credenciais.

## O que foi entregue

Código Python executável, integração com componentes do Pixel-to-Action, percepção OpenCV em CPU, rastreamento de candidatos móveis, identificação inicial manual do jogador, velocidade relativa e aproximação, hash exato e similaridade visual, buffer de eventos, SQLite, workers sem fila crescente, cliente Jev, cliente multimodal remoto, controlador de teclado com trava, replay, diagnóstico de hardware, benchmark sintético, testes e requisitos de aceitação.

O código não treina redes e não modifica pesos. “Aprender” aqui significa registrar hipóteses apoiadas em eventos e recuperá-las quando um trecho visualmente semelhante reaparece. A hipótese não vira verdade porque o modelo afirmou algo com confiança.

## Preparação no Windows

Requisitos iniciais: Python 3.11+, Git, acesso à internet para dependências e clone. O primeiro script procura Python 3.11; se você já usa outra versão compatível, crie `.venv` manualmente com ela antes.

Extraia a pasta e abra PowerShell nela:

```powershell
.\scripts\setup.ps1
```

Não é necessário ativar o ambiente virtual nem alterar permanentemente a política de execução do Windows. Se sua política bloquear scripts, execute os comandos equivalentes abaixo, sem desabilitar proteções do sistema:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,dev]"
.\.venv\Scripts\python.exe scripts\bootstrap_upstream.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ai_player_lab doctor
.\.venv\Scripts\python.exe -m ai_player_lab benchmark
```

O bootstrap clona `danjfarrell/pixel-to-action` em `vendor/pixel-to-action` e fixa o commit `ca697eb9e681a6db657ecc395bb38731960e01f8`. Não sobrescreve uma base existente modificada. O pacote é uma **extensão separada**, não um fork já publicado. Não foi possível executar esse clone no ambiente de construção por ausência de acesso de rede; os arquivos relevantes foram inspecionados via conector GitHub.

O diagnóstico escreve `runs/machine.json`; registra apenas a presença das variáveis de configuração, nunca seus valores ou chaves. O benchmark é local, sem chamadas de API.

## Primeiro teste: apenas observar

Abra o jogo em tamanho fixo, deixe o personagem parado numa tela jogável e rode:

```powershell
.\.venv\Scripts\python.exe -m ai_player_lab calibrate
.\.venv\Scripts\python.exe -m ai_player_lab observe --seconds 120
```

Na calibração, selecione **somente a área do jogo**, confirme e depois selecione o personagem inteiro, incluindo pequena margem. A seleção gera `configs/local.json`. As coordenadas do personagem usam a imagem de trabalho redimensionada, não pixels do monitor.

Mantenha o jogo visível e não cubra sua região com a janela de visualização. A captura é da tela, não de uma aba oculta. O código não acessa memória do emulador nem estado interno do jogo.

**O programa não pressiona teclas neste modo.** Ele apenas registra o que enxerga e a ação proposta por uma política local básica. `floor_y` permanece `null`: a heurística de chão horizontal só é ativada ao definir explicitamente a linha do piso na imagem de trabalho. Ela não é um detector universal de buracos.

Atalhos globais, quando `pynput` estiver disponível:

| Atalho | Ação |
|---|---|
| F6 | Desarma o controle e permite remarcar o personagem |
| F8 | Marca uma morte; guarda contexto anterior, sem inventar sua causa |
| F9 | Arma o teclado somente se houve opt-in e a janela correta está em foco |
| F10 | Desarma e solta as teclas |
| Esc | Encerra a sessão |

Sem hotkeys, `q` fecha a visualização, `d` marca evento e `r` permite remarcar na janela do protótipo. **Depois de morrer/reaparecer, use F6.** Reaquisição automática robusta após morte ainda não está pronta.

## Acrescentar Jev

Configure a variável apenas na sessão do terminal, sem gravar a chave no repositório:

```powershell
$env:TYPESAFE_API_KEY = "SUA_CHAVE"
.\.venv\Scripts\python.exe -m ai_player_lab observe --jev --seconds 60
```

Essa opção faz chamadas reais e pode consumir créditos. Sem `--jev`, nenhuma chamada ao Jev é feita. Padrão: até 3 envios/s e **máximo de 120 chamadas por sessão**. Não há promessa de menor latência do mercado. `max_decision_age_ms` conta desde a captura, incluindo espera e rede; a decisão vencida é descartada. A consulta roda num worker e não interrompe a percepção.

O cliente usa `POST /v1/systemone`, `Choice`, `answers.action.choice` e confiança. O modelo está fixado em `jev-1.14.0`, conforme documentação consultada. O contexto contém dados estruturados, não frames.

## Acrescentar o multimodal por API

**Não instalar llama.cpp, não baixar pesos e não executar modelo local.** O provedor e o modelo multimodal ainda serão escolhidos. O cliente entregue envia `POST /chat/completions`, imagens em `image_url` com data URLs, autenticação Bearer e solicita JSON. A API escolhida precisa aceitar esse contrato; compatibilidade com qualquer provedor não está garantida. Se ela tiver outro formato, será necessário um adaptador.

Configure na mesma sessão do PowerShell:

```powershell
$env:VLM_BASE_URL = "BASE_HTTPS_DO_PROVEDOR"
$env:VLM_MODEL = "ID_DO_MODELO_MULTIMODAL"
$env:VLM_API_KEY = "SUA_CHAVE"
```

Os valores acima são campos para preencher, não um endpoint/modelo selecionado. A base é a URL anterior a `/chat/completions`. Não inclua a chave na URL. Alternativamente, `vlm_base_url` e `vlm_model` em `configs/local.json` têm prioridade sobre as variáveis quando preenchidos; a chave fica exclusivamente em `VLM_API_KEY`. `.env.example` é uma referência, não é carregado automaticamente.

```powershell
# Medição opcional: autoriza três chamadas reais, que podem consumir créditos.
.\.venv\Scripts\python.exe scripts\benchmark_vlm.py --allow-api-calls --runs 3

# Observação com ambas as APIs; continua sem controlar o teclado.
.\.venv\Scripts\python.exe -m ai_player_lab observe --jev --vlm --seconds 60
```

Sem `--jev` e `--vlm`, `observe` e `replay` não fazem chamadas aos modelos. Sem configuração completa, a ativação do multimodal falha com mensagem explícita; não existe fallback para servidor local nem provedor escolhido silenciosamente. O benchmark acima usa as variáveis de ambiente (ou seus argumentos `--base-url`/`--model`), não lê `configs/local.json`.

O multimodal recebe até três imagens reduzidas do recorte do jogo, histórico compacto e evento. Ele não é chamado a cada frame. O worker registra hipóteses, evidências e confiança autorrelatada, não calibrada. JSON inválido é rejeitado. Padrão: intervalo de revisão de 10 s, timeout de 30 s e teto de **12 tentativas por sessão**, contando erros de rede; esses números são limites configuráveis, não desempenho medido nem orçamento monetário.

Mortes registradas continuam no SQLite se a API estiver ocupada ou o teto tiver acabado. A revisão posterior exige novo comando explícito e pode consumir créditos:

```powershell
.\.venv\Scripts\python.exe -m ai_player_lab review-events --limit 3
```

Esse comando tem teto próprio por execução e respeita o menor valor entre `--limit` e `max_vlm_calls`. Reiniciar a sessão renova os limites; não existe limite financeiro global implementado. Configure também um teto de gastos no provedor, quando disponível.

**Dados enviados:** Jev recebe estado estruturado e hipóteses recuperadas; a API multimodal recebe os recortes selecionados e o histórico. A memória SQLite fica local. Não capture chat, senhas ou outras janelas. Sem provedor escolhido, retenção e tratamento de dados ainda precisam ser verificados.

## Controle real: somente depois da inspeção visual

Edite `window_title` e `keys` em `configs/local.json` para corresponderem ao título da janela e aos controles reais do jogo. Evite títulos genéricos como “Chrome”, que podem coincidir com outras abas.

```powershell
.\.venv\Scripts\python.exe -m ai_player_lab observe --jev --vlm --enable-input --seconds 60
```

Começa **desarmado**. Coloque o jogo em foco e pressione F9. Há lista fechada de ações, watchdog, expiração e soltura ao perder foco. F10 e Esc interrompem. Controle real neste protótipo é implementado para Windows; não foi executado no ambiente de construção. Não use em partidas competitivas, sistemas anti-cheat ou jogos que não autorizem automação.

Não há controlador universal de saltos: o reflexo local atual é um **veto conservador de avanço diante de uma borda candidata**, além das travas de validade. Ajustar trajetória de pulo, duração por estado e física precisa de avaliação do jogo.

## Replay e saídas

```powershell
.\.venv\Scripts\python.exe -m ai_player_lab replay caminho\partida.avi --config configs\local.json --headless
```

Replay não pode injetar teclas. Ajuste a calibração ao vídeo. A opção `--jev`/`--vlm`, quando explicitamente fornecida, continua fazendo inferências reais.

Saídas em `runs/<game_id>/`: memória SQLite persistente e sessões separadas com `state.jsonl` e `session.json`. A memória usa as últimas 100 entradas revisadas para busca, com até 12 candidatos verificados geometricamente. Isso limita custo de consulta; não garante reencontro de episódios mais antigos. Logs e evidências contêm somente a região selecionada, mas ainda podem ter dados privados: não publique `runs/`.

## O que os testes NÃO provam

A suite e os vídeos incluídos exercitam contratos, filas, temporalidade, memória, segurança e percepção em imagens sintéticas de aparência fixa. Não provam detecção de todos os sprites, interpretação correta de mortes, latência da API, desempenho da RX 580 nem melhoria de gameplay após tentativas. Consulte `docs/VALIDACAO.md` antes de interpretar os números.

## Documentação

- `docs/REQUISITOS.md`: escopo, requisitos e critérios de aceitação.
- `docs/ARQUITETURA.md`: responsabilidades, contratos, memória e riscos.
- `docs/MODELOS_E_HARDWARE.md`: separação local/API, configuração e o que falta medir.
- `docs/AUDITORIA_UPSTREAM.md`: o que existe na base e o que foi reaproveitado.
- `docs/VALIDACAO.md`: resultados realmente executados e pendências.
- `docs/FONTES.md`: fontes primárias consultadas.
- `NOTICE.md`: proveniência e licença não presumida.
