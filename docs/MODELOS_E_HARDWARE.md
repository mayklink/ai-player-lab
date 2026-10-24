# Modelos por API e hardware local · 0.1.1

## Decisão de Mayk

**Jev e os modelos multimodais serão usados por API remota.** A configuração anterior de SmolVLM2/Qwen em llama.cpp deixa de fazer parte do protótipo. Nenhum peso, runtime de inferência local ou GPU para executar esses modelos é necessário neste perfil.

| Camada | Onde executa | O que precisa ser medido |
|---|---|---|
| Captura e percepção OpenCV | PC de Mayk | CPU, RAM, latência e qualidade visual |
| Rastreamento, mapa visual, memória SQLite | PC de Mayk | IDs, correspondências, custo local |
| Escolha de ações pelo Jev | API remota | RTT, idade da resposta, descarte e consumo |
| Revisão multimodal de eventos | API remota, assíncrona | Qualidade das hipóteses, latência, consumo e tratamento dos dados |
| Validade/reflexos e teclado | PC de Mayk | Ações vencidas, foco e soltura das teclas |

Não há modelo neural obrigatório no ciclo visual rápido. OpenCV não é um modelo multimodal: o caminho implementado usa algoritmos visuais e regras geométricas.

## Configuração das APIs

Jev mantém o cliente existente e `TYPESAFE_API_KEY`. A versão configurada no pacote continua `jev-1.14.0`; esta revisão não muda nem revalida o contrato do serviço.

Para o multimodal, preencher `VLM_BASE_URL`, `VLM_MODEL` e `VLM_API_KEY`. Não foi escolhido um provedor/modelo, e não existe endpoint remoto padrão. Os campos `vlm_base_url` e `vlm_model` do JSON, quando não vazios, têm prioridade sobre o ambiente. A chave não é armazenada em arquivo de configuração.

O adaptador implementado exige HTTPS, Bearer, `POST /chat/completions`, conteúdo de imagem como data URL em `image_url`, retorno textual em `choices[0].message.content` e JSON estruturado validado pelo cliente. Ele solicita `response_format: json_object`. Nem toda API multimodal aceita esse contrato; validar a escolhida ou implementar o adaptador específico.

Não enviar todos os frames: revisões usam até três imagens do recorte, estados e ações resumidos. São enviadas em worker separado. A captura não aguarda o multimodal. A latência do Jev continua sendo uma dependência de jogabilidade; se a resposta chegar velha, o executor a descarta.

## Limites iniciais de chamadas

| Parâmetro | Padrão | Significado |
|---|---:|---|
| `max_jev_calls` | 120 | Teto de tentativas Jev por sessão |
| `jev_hz` | 3 | Frequência máxima de submissão de estado |
| `max_vlm_calls` | 12 | Teto de tentativas multimodais por sessão |
| `review_interval_s` | 10 | Intervalo entre revisões periódicas; mortes são eventos separados |
| `vlm_timeout_s` | 30 | Timeout HTTP multimodal configurado |
| `max_decision_age_ms` | 250 | Validade da ação contada desde o frame de origem |

Metas/limites de engenharia, não medições. Chamadas falhas também podem consumir créditos; no multimodal elas contam no teto e não há retry automático. Eventos não revisados ficam no banco. `review-events` inicia uma execução separada com teto próprio. Não existe orçamento monetário global implementado.

## Máquina de Mayk

Referência informada: Windows e RX 580 de 8 GB. O dispositivo esteve offline na consulta da versão anterior; **não houve nova inspeção remota nem benchmark na máquina nesta revisão**. CPU e RAM atuais continuam sem confirmação.

A capacidade de executar multimodais na RX 580 não é mais um critério de aprovação. Isso não comprova que a máquina suporta a percepção a 30 Hz: ainda precisamos medir captura, CPU, RAM, buffers, compactação de imagens, navegador e controle juntos. Não é necessário discutir CUDA, ROCm ou offload de pesos para esta configuração.

## Plano de validação

1. `doctor` registra hardware local e somente presença das variáveis das APIs, nunca valores das chaves.
2. `benchmark` e `observe` sem flags medem visão local sem chamadas de inferência.
3. Configurar e validar o provedor escolhido; `benchmark_vlm.py --allow-api-calls --runs 3` faz três tentativas reais com fixture sintética. Mede contrato/latência, não compreensão de Mario.
4. Comparar `observe` sem APIs, com `--vlm`, com `--jev` e com ambos: qualidade da percepção, p95 do ciclo, idade das decisões, erros e consumo.
5. Julgar a semântica com episódios reais anotados e comparar mortes repetidas com/sem memória.

Nenhuma chamada autenticada às APIs foi executada nesta revisão. Testes HTTP usam transporte simulado. Valores em reais, taxas e retenção dependem do provedor/modelo que ainda será escolhido; não foram estimados.
