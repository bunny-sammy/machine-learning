# TESTES DE ALGORITMOS DE MACHINE LEARNING
Algoritmos e base de dados fornecidos pelo professor. Adaptações nos códigos foram feitas para abarcar a base de dados.

## Dicionário de dados
A base de dados descreve clientes de uma companhia telefônica e visa classificá-los entre aqueles que cancelaram ou não cancelaram seu plano.
| Variável | Descrição |
|:-------------|:----------------------------------------------------|
| **ID** | Identificação do assinante |
| **Idade** | Idade em anos completos do assinante |
| **Linhas** | Número de linhas do assinante |
| **Temp-cli** | Tempo como assinante em meses |
| **Renda** | Renda familiar do assinante em reais |
| **Fatura** | Despesa média mensal do assinante em reais |
| **Temp_rsd** | Tempo de residência atual do assinante, em anos |
| **Local** | Região onde reside o assinante (A, B, C e D) |
| **Tvcabo** | Assinante possui TV a cabo? |
| **Debaut** | Pagamento em débito automático? |
| **Cancel** | Assinante cancelou o contrato? (**Target** do dataset) |

## Como Começar

### Prerequisitos

* Python 3.x
* pip
* venv

### Instalação

1. Clone o repositório

   ```bash
   git clone https://github.com/bunny-sammy/machine-learning.git
   ```

2. Crie e ative o ambiente virtual

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   # Ou no Windows
   .venv/Scripts/activate
   ```

3. Instale as dependências

   ```bash
   pip install -r requirements.txt
   ```