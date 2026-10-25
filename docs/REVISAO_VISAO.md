# Correção local de visão — 25/10/2026 (não publicada)

Base: `main@9d6534a72bdccc32ca6db29bb46a00ed055cc04a`. A branch foi conferida antes da implementação; não havia mudanças posteriores. Trabalho local em snapshot obtido pelo conector autenticado, sem clone/histórico, push ou PR.

## O que mudou
- **Calibração:** a seleção humana salva um PNG cinza dentro de `player_template`, junto com bbox e tamanho do frame. Inicialização nunca recorta outra aparência na caixa antiga. Metadados, dimensões, textura e PNG são validados. Config com apenas bbox falha antes de clientes/workers; execute `calibrate` novamente.
- **Identidade:** usa aparência original e espelhada, sem adaptação automática de pixels. Picos espacialmente distintos muito semelhantes são recusados. Correlação é score heurístico, limitada a 0,95, não probabilidade calibrada.
- **Reaquisição:** busca local limitada; depois de perda curta, exige duas correspondências espacialmente consistentes. Mais de 1,5 s sem observação aceita ou transição de cena exige F6. Nenhuma busca global que promova outro sprite silenciosamente.
- **Câmera:** HUD excluído antes de selecionar pontos de fluxo óptico.
- **Objetos próximos:** remove mudança do jogador antes/depois do agrupamento morfológico, em vez de descartar todo contorno unido. Slices fora da imagem são ignorados.
- **Persistência:** tracks sem observação podem aparecer como `predicted_motion`, `missed>0`, confiança decrescente, por no máximo 0,2 s (também limitados a cinco frames perdidos). Isso não equivale a detectar um objeto parado.
- **Tempo:** trackers somam tempo realmente decorrido desde a última observação. Replay usa índice/FPS para cinemática; `captured_at` mantém relógio real para TTL e `media_time` registra tempo de mídia. Captura/resize passam a contar na idade live.
- **Bordas:** trecho sem chão até limite é emitido com `right_boundary_visible=False`, `width_is_lower_bound=True`; não se inventa uma margem.
- **Ação mínima local:** para contato iminente (TTC ≤ 350 ms, score ≥ 0,3) veta avanço na direção do objeto. Sobreposição preserva retirada horizontal quando o lado é distinguível; salto neutro no contato é vetado. Não classifica inimigos, não planeja física e esperar pode não evitar dano.

## Migração
Faça nova calibração, mantendo resolução/tamanho fixos. O PNG é um recorte de imagem: não publique seu JSON local. Não altere manualmente apenas bbox ou largura. Se o jogador estiver fora da busca local ao iniciar, use F6.

Configs históricas em `artifacts/` são evidência da versão anterior e permanecem sem migração automática. Para replay sintético novo:

```powershell
.\.venv\Scripts\python.exe scripts\make_synthetic_replay.py --out runs\synthetic-replay
.\.venv\Scripts\python.exe -m ai_player_lab replay runs\synthetic-replay\input.avi --config runs\synthetic-replay\config.json --headless
```

Esses comandos não fazem chamadas aos modelos sem flags explícitas. Não use o helper para fabricar calibração de um jogo real: ele só gera a fixture procedural.

## Verificação executada
- **87 testes passaram:** 69 existentes + 18 casos novos parametrizados.
- Suite nova contra o baseline: **16 falharam, 2 passaram**. Parte das falhas denuncia os novos contratos ausentes; as reproduções comportamentais originais também foram conservadas fora do repositório.
- Casos: fundo na caixa antiga, corrupção/resolução/legado, pose espelhada, ambiguidade, reaquisição curta/longa, HUD, objetos próximos, máscara fora da tela, persistência limitada, buraco aberto/chão contínuo, dt irregular, veto/retirada, relógios de replay e decode artificialmente lento.
- Revisão independente encontrou e fez corrigir duas regressões antes da entrega: slice negativo na máscara anterior e veto que bloqueava retirada no contato.
- Replay integrado MJPG de **75 frames**: 1 frame sem jogador, zero Jev/VLM, teclado desativado. Contagem `accepted` inclui ações `wait`, não significa avanço nem gameplay bem-sucedido.
- Benchmarks de **180 frames** cada, estático e scroll: presença do jogador **98,89%**, erro horizontal médio **0 px** quando detectado. p95 local aproximadamente **3,61/3,82 ms**. Não são números do PC do usuário; não demonstram melhora no jogo.
- Parsing AST de todos os arquivos Python passou.
- Python 3.12.14, OpenCV headless 4.13.0, NumPy 2.3.5, SciPy 1.17.0, HTTPX 0.28.1, pytest 9.0.2, Linux cloud. Dependências de ensaio isoladas, sem mudar requisitos do projeto.

## Limites e riscos restantes
- Não há detector semântico, hitboxes reais, poses gerais, mudança de escala/tamanho, classificação de inimigos ou treino de pesos.
- Uma aparência original/espelhada e duas confirmações não provam identidade. A recuperação é curta, local e conservadora; ambiguidades/transformações ainda exigem remarcação.
- Paralaxe, cenário sem textura, zoom e animação de fundo podem prejudicar a câmera. Excluir HUD não resolve todos esses casos.
- Previsão por 0,2 s pode gerar pausa falsa; depois desse prazo um objeto imóvel deixa de aparecer. É persistência curta, não reconhecimento de objetos estáticos.
- Buraco continua sendo hipótese de uma linha de piso manual; `floor_y=null` desativa essa heurística.
- Replay corrigido assume FPS constante; VFR precisa de PTS por frame. Buffer/revisão e TTL continuam no relógio real.
- Jev continua a 3 Hz e TTL padrão de 250 ms, podendo pausar entre decisões; limites e comportamento pós-orçamento não foram redesenhados.
- Questões preexistentes fora deste núcleo permanecem: recursos de setup podem não fechar se vídeo/upstream/hotkeys falharem antes do loop; falha parcial de injeção pode deixar tecla não registrada como pressionada. **Não habilitar controle real com base apenas nesta revisão.** Teclado real nunca foi acionado.
- Não houve captura Windows, partida real, API real, instalação PowerShell, validação de GPU/PC nem demonstração de redução de mortes.

Próxima validação recomendada: vídeo real anotado, com observação apenas, cobrindo corrida, viradas, saltos, scroll, oclusão e morte/respawn. A entrega é uma correção local testada em laboratório, não homologação do jogador.
