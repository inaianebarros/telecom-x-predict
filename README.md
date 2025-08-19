# Telecom X – Parte 2: Prevendo Churn

O presente projeto tem como objetivo principal prever a evasão de clientes (churn) da empresa Telecom X, por meio da aplicação de técnicas de análise de dados e algoritmos de aprendizado de máquina. A partir da identificação das variáveis que exercem maior influência no cancelamento de serviços, pretende-se construir modelos preditivos capazes de antecipar os clientes com maior probabilidade de evasão, oferecendo suporte estratégico às ações de retenção.

Sua missão consiste em desenvolver modelos preditivos robustos, capazes de indicar, com alto grau de precisão, os clientes mais propensos a cancelar os serviços. Para isso, será necessário estruturar um pipeline de modelagem completo, contemplando desde o tratamento e transformação dos dados até a avaliação comparativa de diferentes algoritmos.

## Objetivos do Desafio 🧠
Preparar os dados para a modelagem (tratamento, encoding, normalização)

- Realizar análise de correlação e seleção de variáveis.
- Treinar dois ou mais modelos de classificação.
- Avaliar o desempenho dos modelos com métricas.
- Interpretar os resultados, incluindo a importância das variáveis.
- Criar uma conclusão estratégica apontando os principais fatores que influenciam a evasão.

## Tecnologias Utilizadas 🛠️
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (preprocessing, model_selection, metrics, modelos)
- imbalanced-learn (SMOTE)
- Jupyter (para executar o notebook)

## Preparação dos Dados ⚙️

### 1. Classificação das variáveis

- **Numéricas:** colunas quantitativas (contínuas ou discretas), identificadas automaticamente no conjunto de dados.

- **Categóricas:** colunas não numéricas (strings ou variáveis categóricas), transformadas por meio de One-Hot Encoding.

### 2. Etapas de transformação

- **Tratamento de valores ausentes:**

  - Variáveis numéricas → substituídas pela mediana.

  - Variáveis categóricas → substituídas pelo valor mais frequente.

- **Normalização:** variáveis numéricas escaladas com StandardScaler, garantindo média 0 e desvio padrão 1.

- **Codificação:** variáveis categóricas transformadas com OneHotEncoder, incluindo o tratamento de categorias desconhecidas (handle_unknown).

### 3. Divisão dos dados

- Treino: 80%

- Teste: 20%

- Aplicada estratificação (no caso de classificação binária) para preservar a proporção original das classes.

### 4. Justificativas das escolhas

- **RandomForestClassifier:** selecionado por sua robustez, capacidade de lidar com variáveis numéricas e categóricas, além de oferecer interpretabilidade via análise de importância das variáveis.

- **Validação Cruzada Estratificada (5 folds):** garante uma avaliação consistente em cenários de classes desbalanceadas.

- **Permutation Importance:** adotado para medir a contribuição de cada variável no modelo final, facilitando a interpretação e a comunicação dos resultados para o negócio.
