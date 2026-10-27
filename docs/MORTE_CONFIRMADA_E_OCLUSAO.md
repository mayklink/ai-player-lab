# Morte confirmada e ocultação breve

O adaptador Mario lê o contador em duas regiões distintas: gameplay e mapa. As referências de dígitos ficam em `profiles/mario/life-glyphs/`, com a origem de cada recorte. Duas leituras válidas e concordantes antes e depois da transição, com decremento de exatamente uma vida, confirmam morte. Leitura sem referência, pontuação baixa, margem insuficiente ou contador instável retorna desconhecido. Só retornar ao mapa não confirma morte.

O episódio fornece essa evidência local ao Luna. Uma classificação incompatível exige uma nova análise, limitada a uma reconsideração. Se o modelo continuar discordando ou estiver indisponível, a morte permanece confirmada e auditável; sua explicação não é promovida a uma estratégia bem-sucedida. A confiança na perda de vida e a hipótese causal são campos separados. Sem plano de ação válido, salva-se o resultado, sem inventar uma solução.

A morte antiga nos canos (`20261024-232700-1`) foi revisada: gameplay x5, depois mapa x4. O frame 10, visualmente já no mapa, tinha sido classificado como desconhecido; a correção histórica registra explicitamente essa anotação manual. A execução ao vivo exige suas próprias duas leituras de mapa. Fontes e resultado corrigido estão em `runs/pipe-death-correction/`.

Durante ocultação parcial, o rastreador pode comparar até 75% dos pixels da máscara, com exigências mais altas de aparência, margem contra inimigos e evidência atual do sprite. Essa correspondência parcial nunca alimenta automaticamente a biblioteca de aparências. Uma posição prevista continua apenas diagnóstica.

Uma ausência muito breve permite executar uma intenção com a última observação real, limitada à janela de 120 ms e leases de no máximo 60 ms. Não são criadas novas coordenadas, não vale em menus ou outra geração, e a ausência prolongada solta as teclas.

O replay temporal usa a mesma thread nativa que solta teclas nos prazos. O envio de comandos não depende da próxima iteração de visão. Cada comando tem no máximo 250 ms e a sequência exige confirmação visual renovada em uma janela de 200 ms. Pausa, interrupção, mudança de cena, divergência ou expiração dessa confirmação cancelam o cronograma. Os horários realmente enviados ficam no trace e no histórico do episódio.
