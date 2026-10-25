# Validação da revisão 0.1.1 · APIs remotas

## Correção local de visão — 25/10/2026, não publicada

Após a revisão abaixo, foram adicionadas 18 regressões e executados **87 testes com sucesso**, além de replay sintético de 75 frames sem API/teclado e benchmarks de 180 frames. Migração obrigatória de calibração, resultados, limites e riscos remanescentes estão em `REVISAO_VISAO.md`. Os artefatos históricos abaixo não representam a nova execução. Nenhuma partida real foi validada.

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
