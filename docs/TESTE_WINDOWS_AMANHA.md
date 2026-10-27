# Teste no Windows — roteiro para amanhã (PR #1 draft)

Objetivo: **validar captura e visão primeiro, sem teclado automático e sem API paga**. Este roteiro não homologa gameplay nem promete FPS na RX 580.

## 1. Conferir código e ambiente
No clone do projeto, preserve qualquer alteração local; se `git status --short` mostrar trabalho não salvo, pare e não force atualização.

```powershell
git status --short
git fetch origin fix/vision-core-20261025
git switch fix/vision-core-20261025
git pull --ff-only
git rev-parse HEAD
```

Se a branch ainda não existir localmente, use `git switch --track origin/fix/vision-core-20261025` no lugar de `git switch`. Anote o SHA exibido; não teste main pensando que contém o PR. Não executar merge/deploy.

Use Python 3.11+ e o ambiente virtual do projeto. Se já preparado:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ai_player_lab doctor
.\.venv\Scripts\python.exe -m pip freeze > runs\windows-dependencies.txt
```

Se ainda não preparado, siga `scripts/setup.ps1`/instalação do README. Não desative proteções do Windows. Guarde Python, dependências, CPU/RAM/GPU/driver, monitor/refresh, escala/DPI, emulador/jogo e filtros/shaders usados. Testes automatizados de teclado usam fakes: passarem não prova injeção real segura.

## 2. Ensaio sintético, sem jogo
```powershell
.\.venv\Scripts\python.exe scripts\make_synthetic_replay.py --out runs\synthetic-replay
.\.venv\Scripts\python.exe -m ai_player_lab replay runs\synthetic-replay\input.avi --config runs\synthetic-replay\config.json --headless
```

Esperado: 75 frames, `jev_calls=0`, `vlm_calls=0`, `real_input_enabled=false`. Presença do jogador não é 100% mesmo na fixture; `accepted` pode ser uma ação `wait`. Se houver exceção, pare antes da captura real e guarde terminal + `session.json`.

## 3. Calibrar a imagem real de novo
Abra apenas o jogo permitido, em tamanho fixo; mantenha o personagem visível. Sem chat/senhas/notificações no recorte:

```powershell
.\.venv\Scripts\python.exe -m ai_player_lab calibrate
.\.venv\Scripts\python.exe -m ai_player_lab observe --seconds 120
```

**Não adicionar `--enable-input`, `--jev` ou `--vlm`.** Jogue manualmente durante a observação. F9 não arma o controlador seco. A janela de overlay não deve cobrir o jogo.

O JSON agora guarda o template real selecionado; não copie apenas bbox de configuração antiga. Ajuste `hud_px` na imagem de trabalho (inteiro ≥ 0 e menor que sua altura), não em pixels do monitor. `floor_y=null` significa sem detecção de buracos. Não invente uma linha de chão para cenas com plataformas/alturas diferentes.

## 4. Sequência manual observável
Anote os horários/frames em que falha. Prefira 5–10 s por caso, sem mudar resolução:

| Caso | O que conferir |
|---|---|
| Parado e pequenas animações | A caixa continua no personagem, não no fundo; score não é probabilidade |
| Corrida e troca de direção | Original/espelho funcionam; poses de corrida não contempladas podem exigir remarcação |
| Oclusão curta durante corrida | Recupera após confirmação; não troca para sprite parecido |
| Oclusão longa, morte e respawn | Deve ficar sem jogador; use F6 para nova marcação, sem reaprender sozinho |
| Início/parada/inversão de scroll | Jogador e objetos permanecem no mesmo referencial; HUD não determina a câmera |
| Objeto perto/encostando | Não desaparecer justamente antes do contato; previsão vem com `missed>0`/`predicted_motion` |
| Objeto que para | Persistência é só até 0,2 s; desaparecimento posterior é limite conhecido |
| Borda com margem fora da tela | Se chão foi corretamente calibrado, candidato deve ter largura mínima/incerta |
| Trocar foco/F10/Esc | Sessão observacional não deve pressionar nada; Esc encerra; isso não mede emergência de teclado real |

Se a caixa grudar no fundo, trocar identidade ou uma transição parecer jogador válido, **pare e guarde o trecho**, não reduza threshold para esconder a falha.

## 5. O que guardar
- SHA, configuração e `runs/<game_id>/<sessão>/session.json` + `state.jsonl`.
- Se disponível, gravação de 60–120 s só do recorte do jogo, sem overlay queimado na entrada, com FPS/timestamps conhecidos.
- Marque 150–300 frames diversos e janelas densas de falha: bbox/centro/pés, pose, visível/ocluído/ausente e identidade. Não são hitboxes físicas.
- O JSON de calibração contém imagem; SQLite/logs/evidências podem ser privados. Não publique `runs/` nem chaves. Compartilhe apenas o recorte necessário.

Replay real exige calibração compatível com a própria gravação. O helper sintético não migra um jogo real e não deve ser usado para inventar sua âncora.

## 6. Ler tempo sem confundir métricas
`session.json.metrics` inclui p50/p95 de captura+resize, visão, **idade do frame ao despachar ação**, idade da decisão e tempo do controlador. Cada linha registra início/fim de captura, resize, visão e despacho. `pixels_changed=false` significa pixels idênticos, não prova captura defeituosa.

- `media_time` é índice/FPS para replay CFR. VFR continua precisando PTS reais por frame.
- `captured_at` e TTL usam relógio monotônico real; atraso de captura/resize e detector de eventos entra no vencimento.
- Esses números não incluem apresentação/scanout nem provam o instante em que o jogo consumiu a tecla.
- Replay sem teclado mede visão/custo, **não** latência tecla → jogo → tela.
- FPS médio alto não elimina frame velho. Registre perdas, troca de identidade, erro p95 e idade p95 por cenário.

Python fornece relógio monotônico para medir intervalos; não misture esse relógio com timestamp de mídia ou hora civil. [Python: time](https://docs.python.org/3/library/time.html#time.monotonic)

## 7. Decisão depois da coleta
1. Se a versão corrigida ainda perde poses reais, primeiro experimentar um banco pequeno de aparências **confirmadas**, em CPU, nos mesmos vídeos anotados. Original+espelho não reconhece automaticamente corrida/salto/transformação; template matching compara um patch com regiões da imagem. [OpenCV: template matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
2. Depois, testar fluxo esparso no jogador entre verificações de aparência, com rejeição de pontos ruins e prazo de expiração; não confundir propagação com reconhecimento. [OpenCV: optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
3. Só comparar backend de captura se idade/custo indicarem gargalo. Não trocar MSS por DXcam ou instalar modelo/driver pesado durante o primeiro ensaio.
4. Jev/VLM continuam opcionais, fora do reflexo local; não mascaram um rastreador errado.
5. Somente após observação real satisfatória, discutir ensaio separado de entrada. O patch reforça limpeza/watchdog e rejeita atalhos do SO, mas não comprova backend Windows. Callbacks do listener mantêm apenas enfileiramento, porque bloquear callbacks pode congelar input no Windows. [pynput: listener thread](https://pynput.readthedocs.io/en/latest/keyboard.html#the-keyboard-listener-thread)

**Critério de saída de amanhã:** um replay/log real reproduzível por cenário e uma lista curta de falhas medidas. Não declarar “joga sozinho” com base nos testes sintéticos.
