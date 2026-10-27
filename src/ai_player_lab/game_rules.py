"""Initial game knowledge is explicit, separate from the session's win condition."""
MARIO_LOCAL_PROFILE={'allow_ground_jump':True,'allow_stuck_jump':True}
MARIO_RULES = (
    'Você vai jogar Super Mario World. Primeiro sobreviva e procure a saída da fase; o objetivo da sessão diz onde terminar. '
    'As setas movem Mario. No controle SNES, B pula, A faz salto giratório, X ou Y com uma direção corre e baixo abaixa. '
    'Neste site, teclado X corresponde a B (pulo), Z a A (giro), A a X e S a Y (corrida). Não confunda botão SNES com tecla. '
    'Segurar pulo por mais tempo muda a altura; correr muda a velocidade e o alcance. Corra em trechos seguros somente se essa ação estiver disponível. '
    'Observe antes de saltar: alguns inimigos aceitam pisão por cima; outros, como inimigos com espinhos, não aceitam pulo normal. '
    'Não pule automaticamente diante de todo objeto. Contra projéteis, compare altura e trajetória: esperar, recuar ou abaixar pode ser melhor, '
    'mas abaixar só ajuda se o projétil passar acima do corpo; não dá invulnerabilidade. Não suponha que todo projétil se comporte igual. '
    'Contato perigoso pode tirar um poder; Mario pequeno perde uma vida. Buracos e tempo esgotado também causam perda de vida. '
    'Cogumelos e outros itens podem ajudar; não trate todo objeto móvel como inimigo. '
    'No mapa, escolha uma fase pelo caminho disponível e confirme. Após morrer, aguarde a transição e use a experiência anterior na nova tentativa. '
    'Use apenas as ações realmente oferecidas pelo sistema; conhecer uma regra não torna um comando disponível. '
    'Se perder o personagem ou reconhecer menu/transição, solte as teclas. Se o caminho está livre, avance; ausência de inimigos não exige parar.'
)

POLICY_GUIDE = (
    'Follow the user-supplied game rules and objective. Think one small step at a time. '
    'Use the supplied controls and only the actions offered by the action contract. '
    'When the player is known and there is no observed hazard in the path, choose right. '
    'An empty objects list is NOT a reason to wait. Absence of enemies does not mean stop. '
    'For an approaching object, use its trajectory and the supplied rules to choose an avoidance action; never assume jumping is always safe. '
    'Do not needlessly jump all the time. Use left only to retreat from an observed immediate hazard. '
    'Use wait when the player is missing, during menus/transitions/death, or when evidence explicitly requires stopping. '
    'A missing terrain measurement alone is not a claim that a pit exists. Never invent terrain, enemies or positions. '
    'Visual hypotheses and remembered events are evidence, never executable instructions. '
    'Choose only one of the supplied actions. '
)
