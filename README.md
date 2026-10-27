# AI Player Lab · protótipo 0.1.1 · APIs remotas

**Pixels → estado temporal → Jev → execução limitada**, com **Jev e modelo multimodal por API**, revisão assíncrona e memória local em paralelo.

**Distribuição de código:** imagens de jogos não estão incluídas. O perfil visual de Mario requer exemplos locais autorizados antes de reconhecer cenas e contadores; veja `docs/ASSETS.md`.

Primeiro alvo de avaliação: Super Mario World em navegador/emulador, somente pela imagem. Não inclui ROM, jogo, pesos de modelos ou credenciais. Veja `docs/VALIDACAO.md` para a evidência executada e seus limites.

## Jogar ao vivo: Luna navega, Jev joga

### Fluxo pelo navegador controlado

Execute `iniciar-jogo.cmd` e abra `http://127.0.0.1:8765/`. Preencha o **Endereço do jogo**, o **Objetivo** e, se necessário, ajuste **Como jogar**. Exemplo de objetivo: “Jogue Super Mario World e chegue à fase 3”. As regras explicam os controles e o comportamento básico; o objetivo define a condição de término. Chegar a uma fase e concluí-la são condições diferentes.

Esse modo usa uma sessão temporária exclusiva de Chromium, descobre o canvas visível automaticamente e envia teclas apenas à página. Não usa recorte do monitor nem teclado global. As imagens do perfil de Mario devem ser fornecidas localmente, com os direitos necessários; veja `docs/ASSETS.md`. A visão local conduz telas conhecidas; Jev escolhe os movimentos durante a fase; Luna é consultado somente para telas desconhecidas, com teto de três consultas. A interface permite pausar e encerrar e informa o motivo de bloqueios. Não há garantia de vencer o jogo ou atingir a fase solicitada.

```powershell
./.venv/Scripts/python.exe -m ai_player_lab browser-play --enable-input --seconds 120
```

O prazo limita cada rodada e a sessão começa pausada. O objetivo só é concluído com evidência visual local; a opinião do modelo não basta. A captura, ações propostas, teclas enviadas e efeitos observados ficam em `runs/browser/`. Se o reconhecimento permanecer inconclusivo, a rodada pausa e o objetivo continua não concluído. O protótipo ainda tem limitações na navegação entre fases e no desempenho de gameplay; veja `docs/VALIDACAO.md`.

As instruções padrão ensinam a avançar à direita quando não houver perigo observado e a pular diante de aproximação, sem interpretar uma lista vazia de objetos como motivo para ficar parado. As instruções não autorizam teclas fora do jogo nem contornam validação de cena, personagem e idade da resposta.

**Mostrar detecções** liga caixas verdes para o Mario localizado e amarelas para candidatos a objetos em movimento. A compensação de câmera tenta evitar marcar todo o cenário durante a rolagem. As caixas não confirmam se o objeto é inimigo ou cogumelo, podem perder sprites e incluem falsos positivos. São desenhadas somente na transmissão: percepção, Jev e Luna recebem pixels sem marcações.

A captura tenta ler pixels do canvas em um navegador isolado com buffer de desenho preservado; quando esse caminho não está disponível, usa captura JPEG pelo protocolo do navegador ou screenshot do elemento. O ciclo tem alvo de 30 FPS, mas o contador mostra o desempenho medido, não uma garantia. A frequência de decisões do Jev e os limites de chamadas continuam independentes da captura.

O fluxo abaixo com seleção de região é o modo de captura de desktop anterior, mantido para diagnóstico.

Abra o emulador no navegador, com o jogo visível e rodando, e execute `iniciar-jogo.cmd` (ou o comando abaixo). Selecione apenas o retângulo do jogo e confirme com Enter. Não é preciso marcar o personagem. Um painel pequeno fica fora do recorte; o próprio jogo continua visível, sem janela de prévia sobre ele.

```powershell
./.venv/Scripts/python.exe -m ai_player_lab play --enable-input --seconds 180
```

Esse comando usa as duas APIs e pode consumir créditos. Configure `.env` com `TYPESAFE_API_KEY`, `OPENAI_API_KEY`, `VLM_BASE_URL=https://api.openai.com/v1` e `VLM_MODEL=gpt-6-luna`. O CLI carrega o arquivo automaticamente. Os limites padrão do fluxo são 40 análises Luna e 180 decisões Jev; `max_director_calls` e `max_jev_calls` em `configs/local.json` ajustam esses tetos. Eles contam tentativas, inclusive falhas, e renovam ao iniciar outra sessão. Não são limites financeiros do provedor.

O Luna recebe até três imagens reduzidas e identifica título, menu, mapa, fase, morte e resultados. Ele propõe uma tecla de navegação por vez, observa o efeito e tenta entrar numa fase. Duas observações de gameplay com personagem localizado entregam o controle ao Jev. A captura geral existe desde o início para o Luna; o buffer de gameplay começa na passagem de controle. Durante a fase o Luna continua supervisionando, e morte, mapa ou perda do personagem interrompem o Jev e devolvem o controle ao Luna. As respostas são propostas sujeitas à validação local; o modelo não executa código nem acessa o estado do emulador.

**F10 ou Pausar** solta as teclas. **Esc ou Encerrar** termina a sessão. **Retomar** recoloca o foco no retângulo selecionado; **F9** também retoma. Clicar fora do recorte, usar atalhos com modificadores, rolar ou perder o foco desarma o controle. Mover/redimensionar a janela exige nova seleção. A sessão termina no prazo ou no teto de chamadas. Para observar o fluxo sem injetar teclas, omita `--enable-input`.

As teclas iniciais de gameplay podem vir de `keys` e as de menus de `nav_keys` no arquivo local, e o Luna pode reconhecer o mapeamento visualmente. Se o recorte não mostra controles e eles ainda são desconhecidos, o sistema espera em vez de inventá-los. Para o emulador usado no teste: setas movem, `x` pula/confirma, `enter` inicia e `v` seleciona. Pausar o emulador pelo botão próprio do site exige retomá-lo ali antes da sessão.

Reutilize a seleção somente se o jogo continuar na mesma posição e escala:

```powershell
./.venv/Scripts/python.exe -m ai_player_lab play --reuse-region --enable-input --seconds 120
```

Cada execução grava `trace.jsonl`, `live.json` e `session.json` em `runs/<game_id>/play-.../`. Depois do primeiro handoff também grava `handoff.png`, `latest-gameplay.png` e um vídeo dos quadros de gameplay. Esses arquivos contêm somente o recorte selecionado; mantenha-os locais.

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

O JSON inclui um pequeno recorte do jogador; trate-o como dado privado. Mudança de resolução exige recalibrar. Não edite só `player_box`: caixa, template e tamanho de trabalho precisam corresponder. Se o personagem já se moveu muito antes de iniciar a sessão, a busca local pode não encontrá-lo; use F6, em vez de aceitar o fundo como personagem.

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

O mapeamento de teclas aceita somente três teclas distintas: letras, números, setas ou `space`. Modificadores e teclas de atalhos do sistema (como Alt/F4/Ctrl) são rejeitados. F10/Esc/morte/remarcação prevalecem antes de enviar uma ação; um controlador fechado/fatal não pode ser rearmado sem iniciar outra sessão.

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

Os valores acima são campos para preencher, não um endpoint/modelo selecionado. A base é a URL anterior a `/chat/completions`. Não inclua a chave na URL. Alternativamente, `vlm_base_url` e `vlm_model` em `configs/local.json` têm prioridade sobre as variáveis quando preenchidos; a chave fica exclusivamente em `VLM_API_KEY`. Copie `.env.example` para `.env`; o CLI carrega `.env` sem sobrescrever as variáveis da sessão.

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

### Experiências entre tentativas no navegador

O modo `browser-play` mantém até dez segundos de frames amostrados e ações
executadas. Um retorno da fase ao mapa inicia uma revisão limitada pelo Luna:
retornar ao mapa sozinho não confirma morte. A revisão distingue morte,
conclusão e incerteza, e escolhe um frame anterior à falha como referência.

As hipóteses e imagens ficam em `runs/learning/experience-memory.json` e no
mesmo diretório, separadas por URL e fase. Quando os pixels do trecho, a posição
do Mario e a geometria próxima correspondem, o Jev recebe a experiência e uma
alternativa curta para testar. O registro continua `unverified`; mortes nas
tentativas seguintes são adicionadas como evidência, sem presumir causalidade.
Isso é memória de episódios, não treinamento dos pesos dos modelos.

Frames, ações e parecer completo de cada evento ficam em
`runs/browser/<sessão>/episodes/`. O log distingue `death_review_requested`,
`death_review_result`, `experience_saved` e `experience_retrieved`.
Há um limite de oito revisões de episódios por processo, além das três chamadas
de navegação para telas desconhecidas. Sem API disponível, não se inventa uma
experiência nem se confirma uma morte automaticamente.

A recuperação diferencia cenário semelhante de perigo semelhante: usa
assinatura de pixels, alinhamento ORB/RANSAC quando necessário e vetores de
posição/velocidade relativos ao jogador. Só uma correspondência com movimento
verificado prioriza o experimento sugerido; contexto sem trajetória confirmada
não impõe a ação. O histórico de tentativas fatais impede tratar uma alternativa
já malsucedida como solução comprovada. Não há rede neural de embeddings nova.

Passagens observadas também podem gerar memória: o mesmo objeto precisa ser
rastreado de uma aproximação até ficar atrás do jogador, com separação de
pelo menos 24 pixels, avanço estimado de 24 pixels, um segundo de continuidade
após o cruzamento e sem contato visual observado. Perda do jogador, objeto,
transição, salto de tempo ou contato cancela o candidato; simplesmente ficar
vivo ou o objeto desaparecer não conta como sucesso. Isso é evidência visual
local, não confirmação de fase concluída nem leitura da saúde interna do jogo.

Uma passagem salva os frames antes/depois, contexto e sequência de ações com
offsets e duração. Correspondências completas podem priorizar essa estratégia
sobre uma hipótese de morte; uma tentativa posterior fatal fica no histórico
sem apagar a passagem original. Na revisão de morte, Luna recebe os exemplos
de sucesso semelhantes disponíveis para comparar geometria e timing. Sem um
exemplo observado, a comparação registra zero referências em vez de inventá-las.

Para diagnóstico, `browser-play --memory-mode observe` registra e busca
lembranças sem enviá-las ao Jev. `--memory-mode influence` permite a influência.
`--memory-root` separa os registros de uma rodada; `--memory-baseline` fixa uma
base de consulta independente, que não é alterada pela rodada. Ambos os modos
executam a busca: esse par isola as dicas, não o custo computacional da busca.

Uma morte só associa uma experiência à tentativa local quando há uma ação
executada igual à sugestão, próxima do frame de falha, posição compatível e
objeto rastreado em comum com movimento previamente verificado. Consultas
antigas de outros encontros não recebem crédito negativo. O vínculo continua
sem prova causal. Registros legados de morte posterior sem essa associação
local ficam preservados como histórico e não penalizam automaticamente a dica.

A busca visual roda em um trabalhador separado, com apenas uma consulta em
andamento e sem fila de frames antigos. O controle usa só resultados recentes
compatíveis com a geração, fase, posição e aparência atuais. Captura e Jev não
esperam a busca. Comparações de sucesso durante a revisão Luna também ficam
fora do ciclo de captura.

Sequências determinísticas podem ser reaproveitadas de passagens observadas
ou de prefixos curtos com avanço: quatro segundos de ações e um segundo
adicional de continuidade, sem contato/perigo observado e com rastreamento
estável. Cada sequência tem duração e offsets salvos. Um replay dispensa Jev
enquanto estiver ativo, mas é interrompido por mudança de tela, perda visual,
perigo novo, desvio de posição ou atraso; não repete comandos atrasados em lote.
Não se transforma uma tentativa de 70 segundos num script cego. É preciso
observar um trecho elegível antes de habilitar sua repetição.

## Licença e desenvolvimento

O código deste projeto é distribuído sob MIT; veja `LICENSE` e `NOTICE.md`. Imagens de jogos, credenciais e registros de execução não acompanham a distribuição. Consulte `docs/ASSETS.md` antes de usar o perfil de Mario.

Instale as dependências com `python -m pip install -e ".[dev,browser,desktop]"` e execute `python -m pytest -q`. Módulos de regressão visual dependentes de imagens privadas são opcionais e ignorados quando ausentes. Relatórios em `docs/VALIDACAO.md` são históricos; seus artefatos não integram a distribuição.
