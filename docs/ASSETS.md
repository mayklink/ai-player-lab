# Imagens opcionais de jogos

A distribuição contém o código de análise, mas não contém imagens, sprites,
ROMs ou recortes de jogos. A licença MIT não concede direitos sobre jogos.

O perfil visual de Mario exige exemplos locais rotulados por uma pessoa.
Sem esses exemplos, o reconhecimento de cenas e contadores fica limitado;
não se deve interpretar a ausência de detecção como confirmação de gameplay.

Quem possuir os direitos necessários pode fornecer seus próprios templates em
`src/ai_player_lab/profiles/mario/`, seguindo os formatos lidos por
`mario_vision.py`, `appearance.py`, `enemy_vision.py` e `life_evidence.py`.
Esses arquivos são ignorados pelo Git e não entram no pacote Python.

Os módulos de regressão visual legados dependem também de imagens locais em
`tests/fixtures/`. Eles são ignorados antes da importação quando a coleção
privada não está disponível. A suite restante usa dados sintéticos e mocks;
aprovação dessa suite não comprova gameplay real. Não baixe imagens ou ROMs
automaticamente para executar testes.
