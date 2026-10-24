# Fontes primárias e histórico de pesquisa

Consulta em 24/10/2026. Especificações de serviços podem mudar; confirmar versão antes de alterar contratos. As páginas abaixo sustentam os fatos externos. Resultados de testes próprios estão nos arquivos de `artifacts/`.

## Base de código

- [Pixel-to-Action — repositório](https://github.com/danjfarrell/pixel-to-action)
- [Commit inspecionado](https://github.com/danjfarrell/pixel-to-action/commit/ca697eb9e681a6db657ecc395bb38731960e01f8)
- [Captura](https://github.com/danjfarrell/pixel-to-action/blob/ca697eb9e681a6db657ecc395bb38731960e01f8/src/pta/capture/screen_capture.py)
- [Contrato de percepção](https://github.com/danjfarrell/pixel-to-action/blob/ca697eb9e681a6db657ecc395bb38731960e01f8/src/pta/perception/base.py)
- [Matcher de templates](https://github.com/danjfarrell/pixel-to-action/blob/ca697eb9e681a6db657ecc395bb38731960e01f8/src/pta/perception/template_match.py)
- [Estado de Zelda](https://github.com/danjfarrell/pixel-to-action/blob/ca697eb9e681a6db657ecc395bb38731960e01f8/src/pta/games/zelda/state.py)
- [Política de Zelda](https://github.com/danjfarrell/pixel-to-action/blob/ca697eb9e681a6db657ecc395bb38731960e01f8/src/pta/games/zelda/policy.py)

Os arquivos foram lidos pelo conector GitHub. O clone no ambiente de construção falhou por indisponibilidade de rede; o script de preparação executa o clone na máquina de destino.

## Visão e inferência local — histórico da versão 0.1

- [OpenCV — fluxo óptico](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html)
- [OpenCV — template matching](https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html)
- [llama.cpp — multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [llama.cpp — compilação e backends](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp — releases oficiais](https://github.com/ggml-org/llama.cpp/releases)
- [SmolVLM2-500M — model card original](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- [SmolVLM2 GGUF — arquivos e tamanhos](https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/tree/main)
- [Qwen3-VL-2B GGUF — arquivos e tamanhos](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/main)
- [AMD — matriz consultada de PyTorch/ROCm no Windows](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)

**Na revisão 0.1.1, os modelos são remotos por decisão de Mayk. As referências de llama.cpp, pesos e ROCm abaixo/acima são históricas, não dependências atuais. Nenhum novo provedor multimodal foi escolhido ou validado.**

A matriz consultada informa que a documentação antiga cobre até ROCm 7.2.1 e aponta unificação posterior da documentação. A ausência de RX 580 nessa matriz não é uma avaliação de todas as implementações comunitárias possíveis; a escolha do protótipo é não depender dessa rota.

## Jev

- [System One — contrato HTTP](https://docs.typesafe.ai/api/system-one)
- [Choice — resposta e probabilidades](https://docs.typesafe.ai/primitives/choice)
- [Confiança — interpretação](https://docs.typesafe.ai/confidence)
- [Modelos e aliases](https://docs.typesafe.ai/models)
- [Imagens — funcionalidade experimental](https://docs.typesafe.ai/images)

## Proveniência

- [GitHub — licenciamento de repositórios](https://docs.github.com/articles/licensing-a-repository)

Nenhuma fonte pública confirma o desempenho desta composição na máquina de Mayk. Essa evidência precisa vir da execução local e dos testes em jogo real.
