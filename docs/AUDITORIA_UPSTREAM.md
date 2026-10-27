# Auditoria do Pixel-to-Action

Repositório: `danjfarrell/pixel-to-action`.
Commit consultado: `ca697eb9e681a6db657ecc395bb38731960e01f8`.
Método: leitura do código pelo conector GitHub; não execução do aplicativo original.

## O que de fato existe

| Componente | Evidência no código | Uso na extensão |
|---|---|---|
| Captura | `src/pta/capture/screen_capture.py`: MSS e array BGR | Reutilizado diretamente em execução ao vivo |
| Matcher | `src/pta/perception/template_match.py`: matchTemplate e caixas | Reutilizado opcionalmente para indicador visual de morte |
| Segmentação | `src/pta/perception/color_mask.py`: HSV e contornos | Não usada como percepção principal |
| Estado Zelda | `games/zelda/state.py`: posição do Link e corações | Não reaproveitado para Mario |
| Política Zelda | `games/zelda/policy.py`: navegação/combate retornam NULL_ACTION | Não tratada como agente jogável |
| Loop | `main.py`: captura, detectores e controlador nulo por padrão | Substituído por runtime independente da extensão |
| Tracking temporal geral | Não consta implementado nos arquivos inspecionados | Acrescentado como protótipo, com limitações explícitas |

## Razão para não executar o entrypoint original

O `main.py` é específico de Windows/RetroArch/Zelda, contém título de janela fixo e detecção ajustada para aquele ambiente. A função de estado divide corações detectados por 16, não pela capacidade real observada de cada partida. O matcher retorna múltiplos pontos acima do limiar sem supressão de sobreposição, podendo supercontar ícones. A política contém stubs de navegação e combate.

Isso não invalida a captura ou o matcher, mas impede apresentar o repositório como um jogador genérico pronto. A extensão usa os componentes existentes úteis e concentra código novo nas partes que faltam: temporalidade, filas, memória, validação de decisões e observabilidade.

## Integração e rastreabilidade

O bootstrap cria um clone em `vendor/`, fixa o commit e não modifica arquivos originais. A extensão carrega `ScreenCapture` pelo caminho do arquivo, sem importar ou executar o `main.py`. O matcher opcional usa as interfaces originais. Testes deste pacote exercitam o código novo; a execução real desses componentes de captura e de entrada no Windows ainda é uma pendência.

O clone não está embutido no ZIP. Não foi publicado fork remoto. A ausência de licença foi registrada em `NOTICE.md`, sem atribuir permissões não concedidas.
