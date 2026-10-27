# Ciclo de tentativas e replay

`attempts.py` mantém, por perfil de jogo e fase, o histórico das teclas realmente enviadas, os horários desde a entrada jogável e o ponto ainda não resolvido. Arquivos em `data/game_profiles/<id>/attempts/` sobrevivem à reinicialização. Objetivos da sessão não entram nessa memória.

Depois de uma morte confirmada, o Luna recebe frames com tempos relativos, a sequência executada e as experiências anteriores daquele plano. Pode propor `retry_plan` com ação, instante e duração limitada a 250 ms. O início anterior à alteração é repetido; a alteração continua explicitamente experimental. Depois, o controle retorna ao Jev. Se um plano realmente executado falha, o mesmo plano não pode voltar a ser aceito como nova experiência sem mudar tempo ou ação.

O replay requer entrada da mesma fase identificada pelo mapa, imagem inicial semelhante e posição inicial compatível. Enquanto está ativo, não são feitas novas chamadas ao Jev. Divergência de posição, perda de rastreamento por mais de 200 ms ou atraso que pule uma mudança de ação interrompem a sequência. Renovações equivalentes da mesma tecla podem ser consolidadas; comandos vencidos não são disparados em rajada.

Uma sequência que sobrevive além do fim da tentativa anterior e avança além do ponto de falha pode ser preservada como passagem local observada uma vez. Isso não prova completar a fase, nem garante sucesso universal. Uma morte posterior encurta o prefixo antes do trecho de falha. A câmera é compensada na estimativa de progresso quando seu movimento é confiável.

O estimador visual agora confirma subida somente depois de uma tecla de salto observada e duas amostras de deslocamento ascendente. Sem essa evidência, mudanças verticais isoladas ficam desconhecidas, em vez de bloquear o reflexo local como se Mario estivesse no ar. Aberturas ou quedas não comandadas ainda exigem interpretação adicional; desconhecido não significa chão confirmado.
