# Perfis persistentes por jogo

O identificador na tela associa URLs ao mesmo jogo. Regras, controles, permissões do controlador e calibração ficam em `data/game_profiles/<id>/profile.json`; experiências e imagens ficam em `learning/` dentro desse diretório. O objetivo permanece apenas na sessão. Esses dados locais não são versionados.

O perfil `super-mario-world` começa com regras e controles conhecidos. Outro identificador começa sem regras, sem controles e sem adaptação visual; criar um perfil não ensina automaticamente a reconhecer outro jogo. O adaptador visual atual continua específico de Mario. Experiências antigas do Mario são copiadas uma única vez, filtradas pela URL, preservando a origem.

O estado temporal informa apoio visual, fase do salto, liberação da tecla, tempo no ar e deslocamento. Apoio exige borda sob os pés e estabilidade em três observações; ausência de evidência aparece como não confirmado. Movimento da câmera e variações na caixa detectada ainda podem afetar essa estimativa.

Para objetos em aproximação, o prazo estimado é o tempo até contato menos a latência recente das decisões e o tempo de subida. Valor negativo significa oportunidade estimada já perdida, não ordem para pular. Sem trajetória ou calibração suficiente, o prazo fica desconhecido. O tempo até subir 16 pixels é atualizado somente após uma tecla de salto observada e deslocamento ascendente; a calibração é salva no perfil.

O Jev recebe esse estado junto à observação. O controlador local evita iniciar os reflexos de salto quando o estado indica que o personagem já está no ar. As medidas são estimativas visuais, não leitura da física interna do emulador.

## Teste real de 24/10/2026

Execução `runs/browser/20261024-223615`: 60,015 segundos, 78 comandos locais, duas mortes classificadas pelo Luna e nenhum erro registrado. Observações enviadas ao Jev incluíram 6 estados apoiados, 5 subindo, 1 no ápice, 6 caindo e 3 desconhecidos. Quatro prazos foram calculados; dois eram negativos. Não houve amostra válida para substituir a estimativa inicial de subida, nem evidência de passagem bem-sucedida do começo. O perfil e a memória foram gravados em `data/game_profiles/super-mario-world/`. Relatório estruturado: `profile-validation.json` no diretório da execução.
