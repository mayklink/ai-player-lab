> Documento histórico: referências a `artifacts/` descrevem evidências antigas que não são distribuídas. Não representam a validação da versão atual.

# Validação da revisão 0.1.1 · APIs remotas

## Registros preservados da main — 25/10/2026

Na revisão noturna original, **120 testes locais passaram**, incluindo ensaios de falha/recuperação com teclado simulado, shutdown e câmera/oclusão. O replay sintético não fez chamadas de API nem enviou input. Não houve teste físico no Windows.

Na correção inicial de visão foram adicionadas 18 regressões e executados **87 testes com sucesso**, além de replay sintético de 75 frames sem API/teclado e benchmarks de 180 frames. A calibração legada precisava ser refeita; os registros e limites daquela revisão estavam em `REVISAO_VISAO.md`. Nenhuma partida real foi validada.

## Execução local de 24/10/2026 — navegador controlado

Atualização de captura e diagnóstico visual: 136 testes passaram. A captura anterior ficou em aproximadamente 8–9 FPS; a leitura de pixels do canvas com buffer preservado mediu aproximadamente 20–29 FPS nos trechos avaliados (incluindo cerca de 20,6 FPS com Mario e um candidato a objeto móvel delimitados na fase). O limite alvo é 30 FPS e não é uma garantia. O caminho JPEG/CDP de fallback mediu aproximadamente 13–15 FPS. A taxa da captura é independente das decisões Jev e do FPS interno do jogo.

O teste visual real registrou caixas no Mario e no objeto que se aproximava. Caixas amarelas indicam movimento, sem identidade semântica confirmada. Os testes também verificam ausência de caixas em fundo imóvel, compensação já existente de câmera e que a cópia anotada não modifica os pixels enviados aos modelos. A visualização usa IDs e timestamps de captura para não contar consultas repetidas como novos quadros.

- 132 testes passaram no Windows do usuário após a inclusão de objetivo, regras iniciais, navegador isolado e regressão visual com um frame real após rolagem da câmera.
- Captura automática do canvas e teclas restritas à página foram executadas no emulador de Super Mario World; a transmissão local ficou em aproximadamente 8–9 quadros/s nas rodadas observadas. Isso não mede o FPS interno do emulador.
- A navegação local passou por título, menus, introdução e mapa e entrou na fase 1 sem chamadas ao Luna nessas telas conhecidas.
- A meta de chegar à fase 3 ainda não foi demonstrada. As rodadas anteriores mostraram mortes, bloqueios e uma perda de reconhecimento após rolagem. Não interpretar testes unitários ou uma ação enviada como prova de objetivo cumprido.
- A perda de reconhecimento foi reproduzida com `tests/fixtures/mario/phase-scrolled.png`. O teste verifica que o HUD com timer positivo e o personagem permanecem reconhecidos sem confundir introdução ou mapa com gameplay.
- Regras iniciais são separadas do objetivo e enviadas ao Jev. Um teste de contrato verifica a instrução explícita de que ausência de inimigos não exige espera; ele não prova qualidade de decisão da API real.

Os relatórios operacionais estão em `runs/browser/` (ignorados pelo Git). A API Jev recebe estado visual estruturado; o fallback Luna recebe somente recortes do jogo. Não houve leitura de RAM do emulador.

## Executado nesta revisão

**69 testes automatizados passaram** no ambiente isolado, Python 3.13.5, OpenCV 4.13.0, NumPy 2.3.5. O relatório está em `artifacts/api-revision/test-results.xml`. Os 51 testes anteriores também foram reexecutados com sucesso antes da alteração.

A suite ampliada verifica configuração por ambiente, ausência de fallback localhost, exigência de endpoint/modelo/chave, HTTPS, proteção contra URL com credenciais, bloqueio de redirecionamentos, contrato de imagens, limite de tentativas, contagem de HTTP 429/timeout e ausência de segredos no diagnóstico. Todo tráfego dos testes de integração usa transporte simulado.

Também foi reexecutado o replay sintético integrado de **75 frames**, sem teclado e com **zero chamadas Jev e zero chamadas multimodais**. Relatório e configuração estão em `artifacts/api-revision/`. Compilação dos módulos Python passou. O ambiente de teste não é o PC de Mayk.

**Não houve chamada real/autenticada a nenhuma API nem escolha de provedor/modelo multimodal.** Não houve partida real, execução no Windows do usuário, inferência local ou download de pesos. Compatibilidade, latência, qualidade e custo das APIs continuam pendentes.

O provedor multimodal precisa aceitar o contrato do cliente ou receber um adaptador próprio. Estes testes não demonstram que qualquer API seja compatível.

## Histórico da versão 0.1

Os dados abaixo foram preservados da entrega anterior. As medições de benchmark sintético não foram recalculadas nesta revisão; a suite e o replay foram reexecutados conforme descrito acima. Os relatórios originais continuam em `artifacts/`.

## Resultado

**51 testes automatizados passaram.** A suite foi executada em Linux isolado, Python 3.13.5, OpenCV 4.13.0, NumPy 2.3.5. Não foi executada no Windows de Mayk. O relatório JUnit está em `artifacts/test-results.xml`.

Cobertura funcional inclui: hash exato e perceptual; correspondência geométrica; deslocamento de câmera; rastreamento do personagem sintético; manutenção/expiração de IDs; aproximação relativa; incerteza de terreno; TTL e episódios; orçamento Jev; contratos HTTP simulados; timeout; schema multimodal; referências de evidência; fila limitada; exceções isoladas; teclado desarmado; watchdog; foco; memória persistente e deduplicação.

## Experimentos sintéticos

Cada benchmark processou 180 frames de 320 × 180, com personagem e objeto procedurais de aparência fixa. Os primeiros cinco frames de cada segmento foram excluídos das estatísticas de tempo como aquecimento. O detector recebe imagens, não posições verdadeiras; as posições conhecidas da fixture servem somente à avaliação e à marcação inicial do personagem.

| Ensaio | Mediana percepção | p95 percepção | Jogador encontrado |
|---|---:|---:|---:|
| Sem rolagem | 2.528 ms | 2.907 ms | 98.89% |
| Com translação de câmera | 2.826 ms | 3.287 ms | 98.89% |

O personagem não foi encontrado em todos os frames. A fixture inclui aproximação/sobreposição de atores, evidenciando que o rastreamento não é infalível mesmo neste teste simplificado. As imagens têm textura e aparência controladas; não representam a variedade de sprites, fundos, animação e paralaxe de Mario World.

Arquivos: `artifacts/synthetic-static/` e `artifacts/synthetic-scroll/`, com JSON, observações e vídeos com overlay. Esses números excluem captura de tela real, inferência de modelos, envio de teclas e jogo. Não converter esses resultados em promessa de FPS na RX 580.

## Replay integrado

Foi executado um replay de 75 frames de vídeo sintético, com política local, memória inicial vazia e controle desativado. Resultado registrado em `artifacts/replay/session-result.json`: 74 ações propostas aceitas pelas travas, 1 frame com jogador não encontrado e zero chamadas ao Jev. A proposta de ação foi somente registrada, não executada em um jogo.

O replay e a suite foram reexecutados depois do ajuste final do agrupamento de movimento. Os ensaios sintéticos e o replay correspondem à percepção entregue. O replay deve ser reexecutado quando mudar percepção, resolução ou configuração.

## Não executado

- Partida real de Super Mario World; nenhum ROM/jogo foi carregado.
- Captura real por MSS/ScreenCapture no computador de Mayk.
- Injeção real de teclas, foco e hotkeys no Windows; testes de controle usaram backend simulado.
- Inferência de SmolVLM2 ou Qwen3-VL; nenhum peso foi baixado.
- Chamada autenticada à API Jev; contratos testados com `httpx.MockTransport`.
- Benchmark de Vulkan/RX 580, CPU ou RAM atuais de Mayk.
- Aprendizado demonstrado por redução de mortes em tentativas sucessivas.
- Instalação PowerShell e clone do upstream no Windows.

## Estado dos critérios de aceitação

G0 parcialmente concluído no ambiente isolado; pendente no PC. G1/G2/G3/G4 dependem de medições reais. Os resultados acima permitem iniciar a avaliação, não declarar o produto pronto.

## Reproduzir

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ai_player_lab benchmark --out runs\synthetic-static
.\.venv\Scripts\python.exe -m ai_player_lab benchmark --scroll --out runs\synthetic-scroll
```
# Reteste de reconhecimento do HUD — 24/10/2026

A máscara de cor dos glifos TIME e do temporizador foi ampliada para tolerar
as bordas alteradas pela compressão JPEG. Mantidos os limites de correspondência
espacial, densidade de tinta, temporizador presente e avaliação de cada frame;
sem retenção indefinida de gameplay. Regressões usam duas capturas reais que
antes eram classificadas como unknown, além de transição preta e intro.

Verificação: 138 testes passaram. Rodada de 60,015 segundos em
`runs/browser/20261024-201039/trace.jsonl`: 78 decisões Jev, 72 aceitas
(60 right, 9 right_jump, 3 wait), 6 rejeitadas por player_missing, zero chamadas
Luna. Nenhuma rejeição por not_stable_gameplay ou old_generation. Houve uma
transição gameplay → loading → map → gameplay, que exige soltar as teclas.
Captura observada durante a fase: 23,5 FPS; após a rodada: 24,4 FPS.
A fase 3 não foi alcançada; Mario ainda encontra dificuldade no obstáculo
inclinado e o rastreamento do personagem apresenta perdas. Este teste valida
a melhora do reconhecimento da fase, não a capacidade de completar o jogo.

## Memória de episódios — 24/10/2026

Implementado buffer amostrado de dez segundos, histórico de ações executadas,
revisão de retorno ao mapa pelo Luna, confirmação/recusa de morte e escolha do
frame anterior à falha. Hipóteses persistem em `runs/learning` e são recuperadas
por URL, fase, pixels, posição do jogador e geometria próxima. O Jev recebe a
alternativa como experimento contextual. Revisões posteriores registram o
resultado da tentativa, mantendo causalidade e sucesso não verificados.

147 testes passaram, incluindo persistência/reabertura, rejeição de outras
cenas/fases/URLs, revisão com evidência rastreável e envio da memória no corpo
real do contrato Jev simulado. Nos testes reais, `20261024-202049` confirmou uma
morte e salvou frames, ações e revisão. `20261024-202302` recuperou a experiência,
mas Jev ainda repetiu o salto; o pedido de decisão foi então ajustado para
explicitar a alternativa acima da regra padrão de avançar.

Reteste final: `runs/browser/20261024-202455/trace.jsonl`, 75,015 segundos.
A experiência `20261024-202302-1` foi recuperada em 89128.171; o pedido Jev em
89128.171 incluiu `suggested_action=wait`, e a decisão seguinte foi `wait`, aceita.
Foram 24 ações aceitas (18 right, 4 right_jump, 2 wait), duas rejeitadas por perda
do jogador e nenhuma nova revisão de morte antes do prazo. O trecho apresentou
perdas de jogador e alternâncias gameplay/unknown; portanto ausência de nova
morte não comprova domínio do obstáculo. Não alcançou a fase 3. A memória e a
mudança pontual de ação foram verificadas; melhora consistente de desempenho
ainda precisa ser demonstrada.

## Assinatura de cenário e movimento relativo — 24/10/2026

Recuperação separa `scene_only` de `scene_and_objects`. Compara posição e
velocidade relativas ao jogador, ignora IDs de rastreamento e ruído de HUD,
rejeita inversão de movimento. Velocidade desconhecida não equivale a zero.
Quando a comparação direta falha, tenta alinhamento ORB/RANSAC com cobertura
espacial, sobreposição, correspondência de jogador e validação por pixels.
Só cena + objetos com movimento verificado permitem priorizar a alternativa
do episódio; cena sozinha é contexto, não prova de recorrência do perigo.

153 testes passaram. Cobertura inclui câmera transladada, texturas distintas,
mesmo vetor com IDs diferentes, velocidades em pixels com ruído, inversão de
movimento e não priorização de uma alternativa seguida de morte posterior.

Teste real de 120,016 s: `runs/browser/20261024-203717/trace.jsonl`.
105 decisões Jev, 93 aceitas, 12 rejeitadas; quatro chamadas Luna, duas mortes
com experiência salva; uma revisão rejeitada por validação (ValueError).
Correspondências completas, com movimento verificado, registradas em
89901.546, 89919.171 e 89940.359. A primeira recuperou
`20261024-203717-2`; a decisão correspondente em 89901.812 executou `jump`,
igual à sugestão aprendida. A tentativa terminou em nova morte: o mecanismo de
reconhecimento/aplicação funcionou, mas a alternativa não foi eficaz.

Após esse resultado, o pedido Jev passou a tratar a alternativa associada a
uma tentativa posterior fatal como falha não causalmente confirmada, sem
priorizá-la como solução. Luna foi instruído a examinar ação e timing já
testados e revisar a hipótese. Essa proteção posterior foi verificada nos
testes automatizados; não foi feita outra rodada ao vivo após ela.
Não chegou à fase 3. Captura ao final: 19,7 FPS. O teste não demonstra que
o agente deixou de repetir o erro nem ganho de latência.

## Reteste da proteção contra alternativa malsucedida

Rodada de 120,015 s em `runs/browser/20261024-204043/trace.jsonl`, resumida em
`protection-validation.json`: 103 chamadas Jev, 94 ações aceitas, nove rejeitadas;
quatro revisões Luna (três mortes e um evento incerto, que não gerou memória).
Nas decisões em 92047.031, 92063.281 e 92103.328, o pedido continha a experiência
`20261024-203717-2` com movimento verificado e
`alternative_failed_in_later_attempt=true`. Jev escolheu left em vez do jump
antigo. Em 92047.406, outra experiência ainda não marcada como falha sugeriu
jump, e Jev executou jump. Não há veto universal ao salto.

As revisões dos episódios 3 e 4 reconheceram explicitamente que a alternativa
de salto já fora tentada e ainda houve morte; propuseram antecipar o salto.
A orientação temporal mudou no texto, mas a ação estruturada continuou jump.
O teste confirma que o feedback chega ao Luna e a proteção muda algumas
decisões Jev; não confirma sobrevivência consistente ou fim da repetição do
erro. O contrato ainda não representa a orientação “mais cedo” como um plano
com condição de distância/tempo e critério de término. Serviço pausou pelo prazo.

## Atribuição local e comparação de influência da memória

Correção: a morte deixa de penalizar todas as memórias consultadas na vida.
Exige ação realmente executada igual à sugestão, janela de -250/+600 ms do
trigger, captura próxima, jogador até 24 px e objeto rastreado em comum.
Só memórias com objeto/movimento verificados são elegíveis. É associação de
experimento local, não prova causal. Entradas antigas sem essa associação não
ativam a penalização. 170 testes passaram, incluindo atribuição a outro objeto,
consulta distante, ação diferente e falta de evidência.

Duas rodadas sequenciais de 120 s, mesma URL/objetivo, navegador isolado novo,
mesmo primeiro frame de pedido Jev e mesmo snapshot imutável. SHA-256 do índice:
`EADB954304B59D40076CB88224CC17B8E5F7338F7C59C572C3156085F594ED2D`.
Registros separados em `runs/ab/20261024-credit/{observe,influence}`;
consulta fixa em `snapshot`. Relatório: `runs/ab/20261024-credit/comparison.json`.

| Métrica | Só registro (213304) | Influência (213541) |
| --- | --- | --- |
| Duração | 120,016 s | 120 s |
| Pedidos Jev | 87 | 39 |
| Ações aceitas / rejeitadas | 76 / 11 | 24 / 15 |
| Ações aceitas | 58 right, 18 left | 15 right, 6 left, 1 right_jump, 1 jump, 1 wait |
| Pedidos com experiência | 0 (12 correspondências em sombra) | 20 |
| Retornos de fase ao mapa revisados | 4 | 0 |
| Revisões Luna | 1 death, 3 uncertain | nenhuma |
| Idade mediana / p95 da resposta | 406 / 542,2 ms | 500 / 663,9 ms |
| Processamento mediano antes do envio Jev | 141 ms | 234 ms |
| Tempo final sem comando Jev | 16,39 s | 75,67 s |
| Passagens verificadas pelo rastreador | 0 | 0 |

No modo influência, a captura mostrou Mario além do encontro inicial, junto ao
obstáculo inclinado, com cinco vidas. Porém houve longa interrupção de controle
e nenhuma passagem validada pelo rastreador. Não classificar isso como sucesso
consistente nem atribuir a diferença às dicas com certeza. No modo observação,
persistiram falhas iniciais sem qualquer experiência no pedido Jev.

Uma rodada por modo, modelo/ambiente não determinísticos e ausência de memória
nova na consulta fixa limitam a conclusão. Ambas executam busca, então esse
teste NÃO isola custo de ORB/recuperação. No braço observação, seis de 17 respostas
com aproximação chegaram com idade maior que o tempo de contato linear estimado
(estimativa visual, não hitbox). O custo do ciclo e a perda do jogador continuam
hipóteses concretas; influência das dicas não explica sozinha a regressão.

## Busca assíncrona, rastreamento e replay determinístico

`LatestMemory` executa no máximo uma busca em thread separada a cada 500 ms,
sem fila. O ciclo usa resultado com idade até 400 ms e revalida geração, fase,
posição, objetos e aparência; continua sem memória se não houver resultado.
Comparações com baselines na revisão de mortes também foram movidas para o
trabalhador de revisão. Testes usam busca bloqueada por evento para provar que
o consumidor segue e não enfileira frames antigos.

O frame bruto que ficou sem controle em 213541 mostrou componente de roupa
vermelha 12×14 com só um pixel azul; a regra exigia dois. Fallback de componente
compacto com contorno escuro/baixa densidade recuperou [150,171,20,24]. Fixtures
reais e negativos cobrem frutas, ausência do jogador, título, intro e mapa.
Não há retenção indefinida de bbox quando o sprite desaparece.

Replay aceita apenas passagem observada ou prefixo observado com avanço,
mantendo offsets/hold_ms e checkpoints de posição, máximo oito segundos.
Prefixos são registrados com quatro segundos de ações e pelo menos um segundo
adicional de continuidade, avanço mínimo de 24 px e sem perigo/contato observado.
Guards interrompem em perda visual, mudança de geração/tela, objeto novo no
caminho, gap, desvio de posição e passo atrasado mais de 150 ms.

Rodadas reais: `runs/browser/20261024-214828` (60,016 s) e `215233` (60,015 s).
Resumo em `runs/async-replay-validation.json`:

| Métrica | Primeira | Segunda |
| --- | --- | --- |
| Pré-envio Jev, mediana | 16 ms | 31 ms |
| Idade da decisão, mediana / p95 | 281,5 / 369,3 ms | 297 / 375 ms |
| Chamadas Jev / ações aceitas | 51 / 46 | 52 / 48 |
| Maior intervalo sem comando em segmentos rotulados gameplay | 1,015 s | 0,672 s |
| Prefixos/passagens verificados e replays ao vivo | 0 | 0 |

No A/B anterior o pré-envio era 141/234 ms e a idade mediana 406/500 ms.
São observações entre rodadas, não benchmark determinístico. Captura observada
em torno de 21–24 FPS; FPS do servidor pausado não representa FPS de decisão.
As novas rodadas ainda perderam o encontro inicial; não chegaram à fase 3.
Replay foi verificado por testes de persistência + execução + interrupção,
mas não ativou ao vivo por falta de trecho aprovado. Não afirmar que já voltou
a passar, que ações remotas operam a 30 Hz ou que o replay resolveu o encontro.
