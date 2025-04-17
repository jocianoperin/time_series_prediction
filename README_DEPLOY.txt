PASSO A PASSO - DEPLOY GPU AUTOMÁTICO PARA VAST.AI (V2.1 CORRIGIDO)

1. Alugue uma instância Vast.ai com SSH habilitado
2. Anote o IP e PORTA SSH da instância
3. Execute no terminal local (Linux ou Git Bash no Windows):

    ./deploy_nuvem_v2.1.sh --ip=<IP> --port=<PORTA> --project-dir=./time_series_prediction

O que esse comando faz:
- Gera chave SSH automaticamente se necessário
- Autoriza o acesso no container remoto
- Sobe o projeto completo para /workspace
- Instala Conda e configura o ambiente `tsenv`
- Ativa corretamente o Conda e executa `main.py`
- Deixa pronto para acesso via VS Code (Remote SSH)

Depois, adicione isso ao ~/.ssh/config:

Host vast-auto
  HostName <IP>
  Port <PORTA>
  User root
  IdentityFile ~/.ssh/id_rsa_vast

E conecte usando: `Remote-SSH: Connect to Host... > vast-auto`
