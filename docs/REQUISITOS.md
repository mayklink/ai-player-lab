# Requisitos de prototipação · 0.1.1 API

## Objetivo verificável

Avaliar se a arquitetura de Mayk consegue obter um estado visual útil, mantê-lo atualizado sem esperar modelos remotos, e usar análise assíncrona de consequências para reduzir erros em um trecho de Super Mario World. O protótipo não precisa concluir o jogo nem generalizar para jogos desconhecidos.

**Hipótese a testar:** separar percepção/controle de baixa latência da revisão semântica permite aprendizado externo por episódios, sem treinamento de pesos e sem colocar o multimodal no caminho crítico. Implementação de software não equivale à confirmação dessa hipótese.

## Escopo inicial e restrições

Um jogo, uma janela visível, resolução/tamanho fixos, câmera predominantemente translacional e um personagem marcado pelo operador. Sem memória do emulador, leitura do motor, dados ocultos, treino de redes ou APIs chamadas a cada frame. Teclado desabilitado por padrão. A base escolhida é Pixel-to-Action, preservada e referenciada por commit.

Cooperação como player 2, jogos 3D, aprendizado de física geral, partidas competitivas, contas, login e exploração automática de qualquer jogo ficam fora desta primeira validação.

**Decisão de implantação:** Jev e o modelo multimodal usam APIs remotas. Captura, visão clássica, rastreamento, memória visual/SQLite e executor permanecem locais. Não há download de pesos, treinamento ou servidor local de inferência no escopo.

## Requisitos funcionais

| ID | Requisito | Implementação entregue | Critério de aceitação em jogo real |
|---|---|---|---|
| RF01 | Capturar apenas a região autorizada | Adapter para ScreenCapture do upstream | 2 min de captura sem incluir desktop fora do recorte |
| RF02 | Transformar frame em estado estruturado | OpenCV, propostas de movimento e rastreadores | Estado contém timestamp, coordenadas, IDs e incerteza; nunca nomeia desconhecido como inimigo |
| RF03 | Acompanhar jogador | ROI inicial + busca local de aparência | Centro dentro de 6 px em >=95% de 200 frames anotados, na resolução de trabalho |
| RF04 | Separar movimento aparente da câmera | Fluxo óptico esparso e deslocamento dominante | Erro mediano <=2 px/frame em 50 pares anotados, ou estado declarado incerto |
| RF05 | Acompanhar objetos móveis | Associação geométrica com expiração | IDs estáveis por >=1 s em >=90% de 20 trajetórias simples; contar trocas de ID |
| RF06 | Estimar aproximação | Velocidade relativa e tempo de contato linear | Erro mediano <=100 ms em 20 contatos visuais, excluindo mudança brusca de trajetória |
| RF07 | Distinguir identidade exata de semelhança | SHA-256 + pHash + ORB/RANSAC | Alteração de pixel muda hash exato; reencontro de trecho exige verificação geométrica |
| RF08 | Consultar Jev fora do loop de percepção | Worker com um item pendente, timeout e orçamento | Atrasar API artificialmente não bloqueia captura; nenhuma ação vencida é executada |
| RF09 | Registrar contexto de consequência | Buffer de 4 s amostrado; F8 ou template de morte | Morte guarda frames anteriores e ações, não apenas tela de game over |
| RF10 | Revisar episódios em paralelo por API remota | Cliente HTTPS com endpoint/modelo configuráveis, schema restrito e evidências | Aceitar apenas JSON validado com referências a frames realmente enviados |
| RF11 | Recuperar experiência anterior | SQLite, similaridade e confirmação geométrica | Em 10 retornos, >=8 matches corretos e nenhum match aceito entre 10 trechos diferentes |
| RF12 | Manter hipóteses, não “verdades” | Status não verificado e confiança autorrelatada | Reprocessar o mesmo evento não aumenta contagem nem confiança automaticamente |
| RF13 | Executar apenas ações permitidas | Wait/left/right/jump/combinações; watchdog | Nenhuma tecla fora do mapeamento; perda de foco e parada liberam controles |
| RF14 | Reproduzir e medir | Replay, logs, métricas p50/p95 | Reproduzir o mesmo vídeo sem acesso ao jogo e obter relatório auditável |
| RF15 | Comparar com e sem memória | Dados necessários registrados | Rodar ablação real; ainda não executada e comparação automatizada pendente |

Os números de aceitação são **metas propostas**, não métricas já obtidas em Mario World.

## Requisitos não funcionais

**RNF01 — Latência.** Meta inicial: captura + percepção + lógica local p95 <=33 ms na máquina de Mayk, a 320 px de largura e alvo de 30 Hz. A rede e o multimodal não podem bloquear esse ciclo. Logs e compactação também entram na medição total; não medir apenas uma função de visão.

**RNF02 — Atualidade.** Decisão usa a hora do frame de origem, não a hora da resposta. Padrão de TTL: 250 ms. Mudança de episódio invalida respostas anteriores. Se a rede exceder o orçamento, o protótipo fica parado; não declara sucesso aumentando o TTL indefinidamente.

**RNF03 — Pressão de fila.** Uma inferência por worker e no máximo um item pendente. Estados antigos são substituídos. Eventos são persistidos mesmo quando sua revisão fica para depois.

**RNF04 — Recursos.** Visão CPU, até dois threads OpenCV no benchmark; sem CUDA exigido. Medir CPU, RAM disponível, captura, buffers e compactação/envio de imagens com o navegador aberto. Não dimensionar RAM/VRAM para pesos de modelos: a inferência foi transferida para APIs.

**RNF05 — Segurança operacional.** Opt-in explícito para entrada real; armação manual; título de janela; limite de duração; teclas liberadas em timeout, erro e perda de foco. O mecanismo não garante que uma aba da mesma janela seja o jogo: usar título específico e supervisionar.

**RNF06 — Privacidade/custo.** Frames somente do recorte. Jev recebe estado textual neste perfil. Multimodal remoto somente após ativação explícita, com recortes selecionados. HTTPS e chave fora de código/logs. Sem download de pesos ou cobrança durante demo/testes. Limites iniciais: 120 tentativas Jev e 12 multimodais por sessão; falhas contam no teto multimodal. Não equivalem a limite financeiro nem persistem entre sessões.

**RNF07 — Rede.** Sem fallback silencioso para inferência local. Falha da API multimodal preserva o evento sem bloquear percepção; resposta Jev vencida não é executada. Medir RTT, idade do estado na decisão e taxa de descarte. Um teto por quantidade de chamadas limita tentativas, não garante custo em reais.

## Portões de validação

**G0 — Preparação:** obter CPU/RAM/driver reais, instalar dependências isoladas, clonar commit da base e passar testes no Windows. Falhou: não avançar para entrada real.

**G1 — Percepção:** gravar 2 min de jogo supervisionado e anotar jogador, movimento de câmera, 20 trajetórias e perigos. Medir qualidade e p95. Falhou: corrigir percepção, não mascarar com mais chamadas de modelo.

**G2 — Semântica:** selecionar 10 episódios, incluindo mortes de causas diferentes e 3 situações ambíguas. Comparar os modelos remotos que forem selecionados em schema válido, evidência correta, reconhecimento de incerteza, latência e custo por episódio. Provedor/modelo ainda não definidos; validar compatibilidade do endpoint com o adaptador antes do ensaio. O modelo deve poder dizer que não há evidência suficiente.

**G3 — Fechamento do ciclo:** ativar Jev em observação, medir idade das decisões e taxa de descarte. Só depois habilitar teclado por 60 s com supervisão.

**G4 — Aprendizado externo:** comparar, no mesmo trecho e condições controladas, 10 tentativas sem memória e 10 com memória, com ordem alternada ou randomizada. Medir mortes repetidas no mesmo perigo, distância progredida, duração e falsas lembranças. Não afirmar “aprendeu” apenas porque a terceira tentativa deu certo.

## Pendências assumidas

Reaquisição após morte ainda manual. Detecção de morte automática depende de template fornecido. Geometria de chão é restrita a piso horizontal calibrado. Sprites que animam, se transformam ou ficam ocluídos podem perder identidade. O protótipo não identifica automaticamente Mario, tartarugas, aliados ou inimigos. Verificar causalidade e generalizar regras permanece uma etapa experimental.
