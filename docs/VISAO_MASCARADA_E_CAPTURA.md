# Captura recente e identidade visual

`LatestCapture` mantém um produtor assíncrono do navegador com um único slot. O consumidor recebe apenas um frame novo e recente; um frame ainda não consumido é substituído, não enfileirado. Navegação e reset invalidam capturas antigas, inclusive as que estavam em andamento. Frames com mais de 150 ms antes do processamento são descartados. Os tempos das observações começam na requisição de captura, incluindo seu custo na idade da decisão.

O OpenCV de classificação e percepção roda fora do loop assíncrono. Isso permite que o navegador capture durante o processamento. O limite de atualização foi elevado para 60 Hz, sem prometer 60 observações processadas por segundo. `capture_fps`, `processing_fps`, `processing_ms`, `frame_age_ms` e `capture_overwritten` distinguem as etapas no status. A exibição usa a mesma observação que alimenta o controle, sem executar outra pipeline de percepção.

O banco `appearance.py` é independente de jogo. Cada perfil guarda recortes positivos, exemplos negativos, máscaras PNG e um manifesto em `appearance_bank/`. A configuração Mario inclui referências explicitamente revisadas em `profiles/mario/identity-seeds/`. Máscaras iniciais conservadoras foram marcadas nos sprites; referências novas usam uma estimativa local de foreground. Essas máscaras não são áreas de colisão do jogo.

Uma revisão da primeira rodada encontrou deriva para nuvens: comparação de aparência sozinha foi insuficiente. O adaptador Mario agora exige também evidência atual de pixels do sprite, incluindo roupa e contraste, tanto na aquisição quanto na busca mascarada. O motor genérico recebe esse validador por callback. Referências aprendidas antes dessa correção foram copiadas para `quarantine-before-semantic-validation/` e retiradas do banco ativo. O caso da nuvem tem teste de regressão mesmo com uma referência incorreta inserida propositalmente.

Uma segunda regressão cobre pernas de inimigo aceitas como pose aprendida. A política de identidade v3 exige evidência de vermelho brilhante da roupa na região superior do recorte. Referências anteriores foram preservadas em `quarantine-before-semantic-validation-v3/`. Esse validador pertence ao adaptador Mario, não ao motor genérico de aparências.

A aquisição combina candidatos de cor com comparação mascarada de aparências completas. Candidatos deslocados corrigem recortes que pegaram só pernas ou fragmentos. Depois da aquisição, a busca fica local à identidade anterior, combinando aparência, rejeição de negativos e limite de deslocamento. Uma posição prevista existe apenas para diagnóstico; ela não é apresentada ao controlador como personagem confirmado. A perda não transfere automaticamente o rótulo para o inimigo.

Novas aparências exigem três confirmações, score alto contra uma referência existente, margem contra negativos, novidade e intervalo de pelo menos 750 ms. Há no máximo 16 referências aprendidas e 24 totais; aprendizado não expulsa as sementes explícitas e os exemplos negativos. O manifesto é substituído atomicamente, e os pixels do jogo continuam sem marcações para os modelos.

## Validação

Testes cobrem isolamento por perfil, persistência, negativos, mudança do fundo fora da máscara, perda de identidade, predição, continuidade e reaproveitamento das detecções. Um conjunto pequeno de sete frames gravados foi revisado e usado no ajuste; não é uma avaliação cega nem estimativa de acurácia geral. O frame 812 forneceu referências adicionais e foi excluído da contagem de sete. Métricas estruturadas ficam em `runs/masked-vision-validation.json`; medições ao vivo ficam em `live-performance.json` no diretório da execução.

Rodada final `20261024-232700`, 60 segundos: mediana de captura 35,6 FPS, processamento 31,5 FPS e idade do frame 62,5 ms nas amostras de fase. Os oito frames distribuídos pela tentativa revisados visualmente mantiveram a caixa em Mario, e houve avanço até os canos verdes. Ainda ocorreram perdas de rastreamento: essas revisões não demonstram ausência de trocas em todos os frames nem conclusão da fase 3. A suíte integrada passou com 286 testes.
