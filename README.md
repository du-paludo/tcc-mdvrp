# tcc-mdvrp

## Estrutura do projeto

tcc_mdvrp/
│
├── data/                  # Arquivos de entrada (instâncias)
│   ├── raw/               # Bases de dados originais
│   └── processed/
│       └── failures/      # Cenários de falhas em JSON gerados por script
│
├── notebooks/             # Exclusivo para Jupyter Notebooks
│
├── src/
│   ├── __init__.py
│   ├── core/              # Entidades do problema
│   │   ├── __init__.py
│   │   ├── entities.py    # Classes: Cliente, Deposito, Veiculo, Rota
│   │   └── solution.py    # Classe que representa uma Solução inteira e calcula Fitness
│   │
│   ├── algorithms/        # Os algoritmos
│   │   ├── __init__.py
│   │   ├── pso.py         # Lógica do enxame (partículas, velocidade)
│   │   └── split.py       # Algoritmo de divisão da rota gigante
│   │
│   ├── utils/             # Ferramentas auxiliares
│   │   ├── __init__.py
│   │   ├── data_loader.py # Lógica para ler os .txt do diretório /data
│   │   └── metrics.py     # Funções para calcular distância euclidiana, etc
│   │
│   ├── scenario/
│   │   └── generate_failures.py  # Gera eventos de falha aleatórios em JSON
│   │
│   └── main.py
│
├── tests/
│   ├── __init__.py
│
├── config.yaml            # Parâmetros do algoritmo (inércia, max_iter, capacidade)
├── requirements.txt       # Dependências (numpy, matplotlib, etc)
└── README.md              # Como rodar o seu projeto


## Descrição

Projeto para estudo de MDVRP com heurísticas (Greedy, GA+PSO), leitura de
instâncias de Cordeau e geração de cenários sintéticos de falhas para
simulações reproduzíveis.


## Pré-requisitos
Certifique-se de ter o seguinte instalado em seu sistema:
- Python 3.8 ou superior
- `pip` para gerenciar pacotes Python

## Configuração do Ambiente

1. **Clone o repositório**:
   ```sh
   git clone <URL_DO_REPOSITORIO>
   cd tcc-mdvrp
    ```
2. **Crie o ambiente virtual**:
   ```sh
   python3 -m venv venv
   ```

3. **Ative o ambiente virtual**:
    - No Windows:
      ```sh
      venv\Scripts\activate
      ```
    - No Linux/Mac:
      ```sh
      source venv/bin/activate
      ```

4. **Instale as dependências**:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

```sh
python3 src/main.py
```

## Animação do log de simulação

Para visualizar a execução da simulação no tempo (rotas, veículos, clientes
visitados e arestas bloqueadas), use o animador de logs.

Comando recomendado (wrapper):

```sh
python3 .\src\tools\animate_simulation_log.py --log-file .\data\processed\simulation_logs\p01_log.json
```

Observação: o wrapper tenta executar automaticamente com o Python da `venv`
(`venv/Scripts/python.exe` ou `.venv/Scripts/python.exe`) quando necessário.

### Exemplo com parâmetros

```sh
python3 .\src\tools\animate_simulation_log.py \
  --log-file .\data\processed\simulation_logs\p21_log.json \
  --fps 30 \
  --speed 8.0 \
  --blocked-edge-ttl 45 \
  --max-blocked-edges 600 \
  --show-ids
```

### Dry-run (sem abrir janela)

```sh
python3 .\src\tools\animate_simulation_log.py --log-file .\data\processed\simulation_logs\p21_log.json --dry-run
```

### Controles de teclado durante a animação

- `space`: play/pause
- `up` / `down`: aumenta/diminui velocidade
- `left` / `right`: volta/avança 5 minutos

### Parâmetros principais

- `--log-file`: arquivo de log da simulação (obrigatório)
- `--instance-file`: caminho explícito da instância Cordeau (opcional)
- `--routes-file`: arquivo de rotas iniciais (opcional)
- `--fps`: taxa de renderização (padrão: `30`)
- `--speed`: velocidade inicial em minutos por segundo (padrão: `8.0`)
- `--start-time`: tempo inicial da simulação (padrão: `0.0`)
- `--blocked-edge-ttl`: tempo de vida visual de arestas bloqueadas (padrão: `45`)
- `--max-blocked-edges`: máximo de arestas bloqueadas desenhadas (padrão: `600`)
- `--show-ids`: mostra identificadores das rotas ao lado dos veículos
- `--dry-run`: valida/parsa os arquivos e imprime resumo sem abrir a animação

## Geração de falhas em JSON

O script [src/scenario/generate_failures.py](src/scenario/generate_failures.py)
gera cenários aleatórios de bloqueio de aresta no formato JSON.

### Exemplo

```sh
python3 src/scenario/generate_failures.py \
  --instance p01 \
  --seed 42 \
  --severity medium \
  --events 3 \
  --max-time 120.0
```

Saída padrão:

```text
data/processed/failures/p01_seed42.json
```

### Formato gerado

```json
{
  "metadata": {
    "instance": "p01",
    "seed": 42,
    "severity": "medium",
    "generated_at": "2026-04-12"
  },
  "events": [
    {
      "trigger_time": 15.5,
      "type": "edge_block",
      "node_a": 5,
      "node_b": 12
    }
  ]
}
```

### Parâmetros principais

- `--instance`: instância Cordeau usada como base (`p01`, `p23`, etc.).
- `--seed`: controla reprodutibilidade do sorteio.
- `--events`: número de eventos de falha gerados.
- `--max-time`: limite superior para sorteio de `trigger_time`.
- `--severity`: rótulo de cenário (`low`, `medium`, `high`) salvo em metadata.
- `--output`: caminho final do JSON (opcional).
- `--data-file`: caminho explícito para arquivo de instância (opcional).

Observação: no estado atual, `severity` é metadado de cenário e não altera
automaticamente o número de eventos ou a distribuição temporal.