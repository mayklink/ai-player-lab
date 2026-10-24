# Arquitetura implementada

```text
ScreenCapture do Pixel-to-Action / vídeo de replay
    ↓ recorte escolhido + redução de resolução
OpenCV: deslocamento de câmera + diferenças + jogador marcado
    ↓
Associação temporal → IDs, caixas, velocidades, aproximação, hipóteses de chão
    ↓ estado com timestamp e episódio
    ├── worker → API Jev → ação curta → validade/TTL → veto local → teclado opt-in
    ├── buffer temporal → evento → SQLite → worker → API multimodal
    │                                      ↓
    │                              hipótese + evidência + confiança
    └── worker de busca visual ← memória de episódios revisados
```

## Implantação

Jev e multimodal são serviços remotos. O processo local mantém captura, visão, associação temporal, memória e teclado; não hospeda modelos. O protocolo multimodal entregue é configurável, mas não universal. Não há endpoint/modelo padrão nem execução remota sem flags explícitas. O `review-events` e o benchmark autorizado são comandos separados que também fazem chamadas reais.

## Papel das duas camadas propostas por Mayk

O ciclo rápido não precisa compreender narrativamente cada imagem: mantém uma estimativa visual limitada e identifica quando essa estimativa deixou de ser confiável. O ciclo lento examina sequências e consequências. Se ele demora, o ciclo rápido continua; o conhecimento chega depois.

A revisão não escreve diretamente no teclado e não altera pesos. O Jev recebe um pequeno conjunto de hipóteses recuperadas por contexto. A mesma estrutura permite substituir o multimodal sem reescrever captura, rastreamento ou executor.

## Identidades distintas

Uma ocorrência de frame é identificada por sessão, episódio, sequência e timestamp. `frame_hash` identifica conteúdo exato para auditoria/deduplicação. SHA-256 é uma assinatura prática, não uma demonstração matemática de ausência de colisões.

`scene_hash` é uma assinatura perceptual da imagem com regiões dinâmicas conhecidas mascaradas. Serve para ordenar candidatos; não basta para afirmar que dois trechos são iguais. ORB e verificação geométrica testam correspondências. Ambientes repetidos, pouco texturizados ou com paralaxe podem falhar. Não há mapa global nem localização perfeita.

IDs de rastreamento valem localmente; não significam que o objeto continua sendo o mesmo depois de sair da tela ou reaparecer após a morte.

## Percepção e tempo

As velocidades são medidas em pixels da imagem de trabalho por segundo. O deslocamento dominante da câmera ajuda a separar mudanças do cenário. Velocidade relativa entre personagem e objeto cancela uma translação comum, mas não resolve paralaxe, hitboxes invisíveis ou objetos que mudam de direção.

Propostas de movimento são regiões candidatas: uma mudança de animação pode criar várias propostas para um objeto, e dois objetos encostados podem se fundir. Objetos imóveis não são descobertos por esse detector. O jogador recebe uma marcação inicial manual para evitar fingir que o sistema já sabe reconhecê-lo.

## Reflexos e executor

O controlador implementado não é uma IA geral de reflexos. Ele bloqueia ações vencidas, de outro episódio, sem jogador confiável ou de baixa confiança. Também pode vetar avanço direto sobre uma lacuna candidata próxima. Um executor separado mantém as teclas por tempo limitado, libera ao perder foco e exige armação manual.

Planejamento de trajetória de salto, antecipação de aterrissagem e descoberta automática da física ainda não estão implementados. Não chamar esses componentes de resolvidos.

## Revisão de eventos

F8 é a referência manual inicial de morte. O sistema guarda até 4 s de observações amostradas; escolhe até 3 imagens e uma âncora anterior ao evento. Opcionalmente, um template de tela/indicador de morte usa o matcher original do Pixel-to-Action.

O multimodal recebe frames, ações e estado resumido. Sua saída deve conter uma hipótese, evidências existentes e ação sugerida da lista fechada. A confiança é autorrelatada e não equivale à probabilidade de a regra estar correta. Dados observados e explicações inferidas permanecem separados.

A busca posterior exige correspondência visual do lugar. Uma lembrança antiga não é convertida automaticamente em coordenada válida na tela atual nem em comando obrigatório. Esse cuidado evita reutilizar um retângulo antigo depois de a câmera andar.

## Falhas previstas

API lenta: descartar resultado vencido; não bloquear captura. Multimodal ocupado: preservar evento e revisar posteriormente. JSON inválido: rejeitar. Cena sem textura: não confirmar localização. Personagem perdido: parar ações e remarcar. Troca de episódio: invalidar decisões anteriores. Mudança de resolução: recalibrar. Rede indisponível: manter captura, preservar eventos e rejeitar decisões vencidas. Teto de chamadas atingido: não enviar novas inferências; mortes registradas continuam pendentes. Custo e retenção: verificar no provedor escolhido antes de enviar imagens.
