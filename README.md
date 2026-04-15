# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
  <a href="https://www.fiap.com.br/">
    <img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Administração Paulista" border="0" width=40% height=40%>
  </a>
</p>

<br>

# Flexmedia — Sistema de Análise de Comportamento em Totem de Loja

## Grupo Solo

## 👨‍🎓 Integrantes:
- Vitório Stevanatto Compri Paciulo — RM567895


## 📜 Descrição

O **Flexmedia** é um sistema de análise de comportamento de usuários em totens interativos de loja, desenvolvido como Enterprise Challenge da FIAP. O totem possui três telas navegáveis — **INÍCIO**, **CATÁLOGO** e **PROMOÇÕES** — e registra, via ESP32 (simulado no Wokwi), o tempo que cada usuário permanece em cada tela e a sequência de navegação entre elas.

O sistema realiza a ingestão dos logs seriais do ESP32, armazena os eventos em banco de dados Oracle (FIAP) e em CSV local, e aplica análise estatística, Machine Learning e Inteligência Artificial para extrair padrões de comportamento e recomendar conteúdo ao usuário em tempo real.

**Funcionalidades principais:**

- **Ingestão de logs** do totem ESP32 com validação e deduplicação
- **Análise estatística** completa: distribuição de tempo por tela, padrões por hora do dia, heatmaps de engajamento
- **Machine Learning** com Random Forest: classificação de engajamento (ALTO/BAIXO) e predição da próxima tela com métricas formais (F1, precisão, recall, validação cruzada)
- **Recomendação via Cadeia de Markov**: aprende padrões de navegação e sugere a próxima tela em tempo real
- **Chat IA integrado**: assistente em linguagem natural que responde perguntas sobre os dados do totem
- **Dashboard interativo** em Streamlit com 5 abas unificadas
- **Simulação de visão computacional** via coluna `presence_detect` (detecção de presença)

O sistema foi evoluído ao longo de 4 sprints, partindo de uma prova de conceito simples (Sprints 1 e 2) até uma plataforma completa com IA embarcada (Sprints 3 e 4).

---

## 📁 Estrutura de pastas

```
Flexmedia/
│
├── assets/                      # Imagens e recursos visuais
│   ├── logo-fiap.png
│   └── graficos/                # Gráficos gerados automaticamente
│       ├── 01_boxplot_dwell.png
│       ├── 02_acessos_por_tela.png
│       ├── 03_tempo_medio_tela.png
│       ├── 04_acessos_por_hora.png
│       ├── 05_heatmap_hora_tela.png
│       ├── 06_engajamento_por_tela.png
│       ├── 07_feature_importance_engajamento.png
│       ├── 07_feature_importance_proxima_tela.png
│       ├── 08_confusion_matrix_engajamento.png
│       ├── 08_confusion_matrix_proxima_tela.png
│       └── 09_matriz_transicao.png
│
├── config/                      # Configurações e credenciais
│   ├── config.py                # Carrega variáveis de ambiente
│   └── .env.example             # Template de credenciais (nunca commitar o .env real)
│
├── document/                    # Dados e documentação
│   ├── dados_simulados.csv      # Dataset gerado (268 eventos, 35 sessões)
│   ├── logs_telas.txt           # Log serial bruto do ESP32
│   └── other/
│
├── scripts/                     # Scripts auxiliares
│
├── src/                         # Código-fonte principal
│   ├── dashboard_flexmedia.py   # Dashboard Streamlit (5 abas)
│   ├── analise_estatistica.py   # Análise estatística + 6 gráficos
│   ├── modelo_ml.py             # Random Forest (engajamento + próxima tela)
│   ├── recomendacao.py          # Cadeia de Markov para recomendação
│   ├── gera_dados.py            # Geração de dataset simulado realista
│   └── importa_logs_oracle.py   # Ingestão de logs no Oracle com validação
│
├── README.md
└── requirements.txt
```

---

## 🔧 Como executar o código

### Pré-requisitos

- Python 3.10+
- Acesso ao Oracle FIAP (`oracle.fiap.com.br:1521/ORCL`)
- Git

### 1. Clonar o repositório

```bash
git clone https://github.com/viscp1011/Flexmedia.git
cd Flexmedia
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar credenciais Oracle

Crie um arquivo `.env` na raiz baseado em `config/.env.example`:

```
ORACLE_USER=seu_rm
ORACLE_PASSWORD=sua_senha
ORACLE_HOST=oracle.fiap.com.br
ORACLE_PORT=1521
ORACLE_SID=ORCL
```

### 4. Gerar dados simulados (opcional — sem Oracle)

```bash
python src/gera_dados.py
```

### 5. Importar logs reais do ESP32 (opcional)

```bash
python src/importa_logs_oracle.py --file document/logs_telas.txt
```

### 6. Rodar análise estatística e ML

```bash
python src/analise_estatistica.py
python src/modelo_ml.py
python src/recomendacao.py
```

Os gráficos são salvos automaticamente em `assets/graficos/`.

### 7. Iniciar o dashboard

```bash
streamlit run src/dashboard_flexmedia.py
```

Acesse `http://localhost:8501`. O sistema conecta ao Oracle automaticamente; caso indisponível, usa o CSV local como fallback.

---

## 🗃 Histórico de lançamentos

* **0.4.0 — Sprint 4 (Abr/2025)**
  * Cadeia de Markov para recomendação de próxima tela em tempo real
  * Chat IA integrado ao dashboard com NLP em português
  * Simulação de visão computacional (detecção de presença)
  * Dashboard unificado com 5 abas

* **0.3.0 — Sprint 3 (Mar/2025)**
  * Análise estatística completa com Matplotlib/Seaborn (6 gráficos)
  * Dois modelos Random Forest com métricas formais e validação cruzada
  * Gerador de dados simulados com comportamento realista
  * Segurança: credenciais via variáveis de ambiente (python-dotenv)

* **0.2.0 — Sprint 2 (Fev/2025)**
  * Importação de logs do ESP32 para Oracle com validação
  * Análise exploratória inicial dos dados
  * Nota: 8.2/9.0

* **0.1.0 — Sprint 1 (Jan/2025)**
  * Prova de conceito: simulação do totem ESP32 no Wokwi
  * Estrutura inicial do repositório

---

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/viscp1011/Flexmedia">FLEXMEDIA</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Vitório Stevanatto Compri Paciulo — FIAP</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
