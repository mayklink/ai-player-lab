# Alterações

## Correções da main recuperadas — 25/10/2026

- Press/release parcial é rastreado para limpeza; teclas cuja liberação falhou não são esquecidas. Controlador fechado/fatal não rearma e atalhos de SO não são aceitos como controles.
- F10/Esc/morte/remarcação prevalecem antes do envio de ação; TTL é rechecado depois do detector de eventos.
- Cleanup incremental cobre falhas de setup. Shutdown descarta jobs pendentes e fecha o provider após terminar trabalho já em voo.
- Reaquisição correndo usa continuidade de movimento; observações/predições compartilham compensação e filtro de câmera. HUD inválido é rejeitado.
- Logs incluem idade do frame/decisão e timestamps por estágio; roteiro Windows e pesquisa orientam observação real antes de modelos/teclado.
- Calibração persiste aparência PNG e valida dimensões; templates rejeitam ambiguidade e perda longa/transição exige F6.
- Tracks previstos ficam explicitamente incertos por até 0,2 s; o tempo real acumulado após oclusão é usado.
- Buraco aberto conserva largura mínima; existe veto local direcional de contato iminente.
- Replay separa tempo de mídia de TTL; helper cria replay sintético com calibração compatível.
- A revisão original tinha 120 testes e a correção de visão 87 regressões. A validação atual, após a recuperação, passou em 377 testes; nenhuma publicação foi feita.

## 0.1.1 — Jev e multimodal por API

- Mantido Jev remoto; alterado multimodal para configuração explícita de API HTTPS, modelo e chave.
- Removidos endpoint localhost padrão, alias `local-vlm` e script de início/download de modelos locais.
- Adicionados limite multimodal por execução, contagem de tentativas com falhas, bloqueio de redirects e validação de configuração.
- Captura, percepção, memória e controle continuam locais; observação sem flags continua sem chamadas de inferência.
- Atualizados requisitos, arquitetura, hardware, instruções, diagnóstico e testes. Benchmark remoto exige autorização explícita.
- Provedor/modelo multimodal não selecionados. Integração autenticada e gameplay real ainda não executados.

O arquivo 0.1 original foi preservado. Relatórios sintéticos herdados em `artifacts/` são históricos; a nova suite está em `artifacts/api-revision/`.
