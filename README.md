# Trabalho II – Comunicação Digital

Simulação de um sistema de comunicação digital completo, desde a geração da mensagem até a recuperação dos dados, passando por codificação de canal, modulação e análise de desempenho.

## O que faz

O programa simula a transmissão de uma mensagem de texto através de um canal com ruído, usando:

- Conversão de texto ASCII para bits
- Codificação Manchester (cada bit vira 2 bits)
- Modulação BPSK ou QPSK
- Adição de ruído AWGN no canal
- Demodulação e decodificação
- Cálculo da taxa de erro de bits (BER)

Ao final, gera um gráfico comparando o desempenho de BPSK e QPSK em diferentes valores de SNR.

## Como rodar

Primeiro, crie um ambiente virtual (recomendado - talvez voce precise utilizar python3 nesse comando):

```bash
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

Depois, instale as dependências:

```bash
pip install -r requirements.txt
```

E execute:

```bash
python -m src.main
```

Os resultados aparecem na pasta `output/`:
- `ber_results_bpsk_qpsk.txt` - tabela com os valores de BER
- `ber_curve_bpsk_qpsk.png` - gráfico comparativo

## O que acontece durante a execução

O programa mostra no console cada etapa do processo:

Para cada valor de SNR testado, você vê quantos erros ocorreram e como ficou a mensagem recebida (mesmo que com erros).

Resultados obtidos pelo grupo ja estao nas pastas `output`.

## Personalizar

Para mudar a mensagem ou os valores de SNR, edite a função `main.py` na pasta `src`:

## Estrutura do código

- `src/main.py` - ponto de entrada, chama a simulação
- `src/encoding.py` - conversão texto/bits e codificação Manchester
- `src/modulation.py` - modulação BPSK e QPSK
- `src/channel.py` - adiciona ruído AWGN
- `src/simulation.py` - orquestra tudo e calcula o BER
