# Alterações

## 0.1.1 — Jev e multimodal por API

- Mantido Jev remoto; alterado multimodal para configuração explícita de API HTTPS, modelo e chave.
- Removidos endpoint localhost padrão, alias `local-vlm` e script de início/download de modelos locais.
- Adicionados limite multimodal por execução, contagem de tentativas com falhas, bloqueio de redirects e validação de configuração.
- Captura, percepção, memória e controle continuam locais; observação sem flags continua sem chamadas de inferência.
- Atualizados requisitos, arquitetura, hardware, instruções, diagnóstico e testes. Benchmark remoto exige autorização explícita.
- Provedor/modelo multimodal não selecionados. Integração autenticada e gameplay real ainda não executados.

O arquivo 0.1 original foi preservado. Relatórios sintéticos herdados em `artifacts/` são históricos; a nova suite está em `artifacts/api-revision/`.
