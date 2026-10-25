# Alterações

## Não publicado — correção local de visão, 25/10/2026

- Calibração persiste aparência PNG e valida dimensões; configurações legadas exigem recalibrar.
- Template original/espelhado, rejeição de ambiguidade e reaquisição curta com confirmação; perda longa/transição exige F6.
- HUD excluído da câmera; máscara do jogador aplicada antes de agrupar movimentos próximos.
- Tracks previstos explicitamente incertos por até 0,2 s; tempo real acumulado após oclusão.
- Buraco aberto representado com largura mínima, não descartado; veto local direcional de contato iminente.
- Replay separa tempo de mídia de TTL; helper cria replay sintético com calibração compatível.
- Testes de regressão locais e documentação de limites; nenhuma publicação, API real ou controle real executado.

## 0.1.1 — Jev e multimodal por API

- Mantido Jev remoto; alterado multimodal para configuração explícita de API HTTPS, modelo e chave.
- Removidos endpoint localhost padrão, alias `local-vlm` e script de início/download de modelos locais.
- Adicionados limite multimodal por execução, contagem de tentativas com falhas, bloqueio de redirects e validação de configuração.
- Captura, percepção, memória e controle continuam locais; observação sem flags continua sem chamadas de inferência.
- Atualizados requisitos, arquitetura, hardware, instruções, diagnóstico e testes. Benchmark remoto exige autorização explícita.
- Provedor/modelo multimodal não selecionados. Integração autenticada e gameplay real ainda não executados.

O arquivo 0.1 original foi preservado. Relatórios sintéticos herdados em `artifacts/` são históricos; a nova suite está em `artifacts/api-revision/`.
