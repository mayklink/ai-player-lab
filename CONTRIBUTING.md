# Contribuir

Use Python 3.11 ou superior e um ambiente virtual dedicado.
Instale `python -m pip install -e ".[dev,browser,desktop]"` e execute
`python -m pytest -q` antes de enviar alterações.

Não inclua `.env`, chaves, capturas pessoais, bancos de dados, vídeos,
diretórios de execução ou imagens de terceiros sem licença de redistribuição.
Use `.env.example` para documentar nomes de configuração sem valores secretos.
Testes devem usar mocks para chamadas de API e não enviar teclas globais.

Descreva a mudança, como foi validada e suas limitações. Contribuições de
código devem ser compatíveis com a licença MIT do projeto.
