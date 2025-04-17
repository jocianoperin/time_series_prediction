PASSO A PASSO - DEPLOY GPU AUTOMÁTICO PARA VAST.AI (V3)

1. Edite o arquivo deploy/config.env:
    IP=<IP_DA_INSTANCIA>
    PORT=<PORTA_SSH>

2. Execute no terminal local:

    cd deploy
    ./deploy_nuvem.sh

O que será feito:
- Geração e envio automático da chave SSH
- Envio do projeto para /workspace/time_series_prediction
- Instalação do Conda
- Criação do ambiente tsenv com environment.yml ou requirements.txt
- NÃO roda o main.py automaticamente

3. Acesse via VS Code:
    Adicione em ~/.ssh/config:

Host vast-auto
  HostName <IP>
  Port <PORTA>
  User root
  IdentityFile ~/.ssh/id_rsa_vast

E conecte com Remote-SSH: Connect to Host... > vast-auto

4. Para baixar predições:
    ./baixar_pastas.sh

5. Para enviar dados:
    ./enviar_dados.sh
