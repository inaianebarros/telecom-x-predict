# Telecom X – Parte 2: Prevendo Churn

O presente projeto tem como objetivo principal prever a evasão de clientes (churn) da empresa Telecom X, por meio da aplicação de técnicas de análise de dados e algoritmos de aprendizado de máquina. A partir da identificação das variáveis que exercem maior influência no cancelamento de serviços, pretende-se construir modelos preditivos capazes de antecipar os clientes com maior probabilidade de evasão, oferecendo suporte estratégico às ações de retenção.

Sua missão consiste em desenvolver modelos preditivos robustos, capazes de indicar, com alto grau de precisão, os clientes mais propensos a cancelar os serviços. Para isso, será necessário estruturar um pipeline de modelagem completo, contemplando desde o tratamento e transformação dos dados até a avaliação comparativa de diferentes algoritmos.

## Objetivos do Desafio 🧠

- Preparar os dados para a modelagem (tratamento, encoding, normalização).
- Realizar análise de correlação e seleção de variáveis.
- Treinar dois ou mais modelos de classificação.
- Avaliar o desempenho dos modelos com métricas.Interpretar os resultados, incluindo a importância das variáveis.
- Criar uma conclusão estratégica apontando os principais fatores que influenciam a evasão.

## Tecnologias Utilizadas 🛠️

- pandas, numpy
- matplotlib, seaborn
- scikit-learn (preprocessing, model_selection, metrics, modelos)
- imbalanced-learn (SMOTE)
- Jupyter (para executar o notebook)

## O que você vai praticar 🧰

- Pré-processamento de dados para Machine Learning
- Construção e avaliação de modelos preditivos
- Interpretação dos resultados e entrega de insights
- Comunicação técnica com foco estratégico

## Tratamento dos Dados ⚙️

Principais etapas realizadas:

- Importação e tratamento de dados, incluindo remoção de variáveis irrelevantes
- Análise de correlação para seleção e exclusão de variáveis redundantes.
- Reparação dos dados para modelagem preditiva.

## Encoding 🔡

No conjunto de dados da Telecom X, diversas colunas apresentavam valores categóricos, como por exemplo:
account.Contract', 'account.PaymentMethod' e 'internet.InternetService'

Como os algoritmos de aprendizado de máquina trabalham apenas com valores numéricos, foi necessário aplicar técnicas de encoding:

- **Label Encoding:** aplicado em variáveis binárias (ex.: Yes/No → 1/0).

- **One-Hot Encoding:** aplicado em variáveis com múltiplas categorias, criando novas colunas para cada possível valor.

## Verificação da Proporção de Evasão 📊 

Calculamos a proporção de clientes cancelados (evadidos) e os que permaneceram. A distribuição será usada para definir se o dataset está balanceado.

- Equilíbrio: Quando as classes estão próximas de 50% / 50%.

- Gerenciável quando próximo de 70% / 30% .Se 70% dos clientes permaneceram e 30% cancelaram, ainda é possível treinar bons modelos com técnicas de balanceamento

- Desequilíbrio forte: Quando uma classe tem 80% ou mais dos registros (ex: 80% "Não" e 20% "Sim"), o modelo pode ignorar a classe minoritária, e será importante:

- Aplicar técnicas de balanceamento (como oversampling com SMOTE, undersampling, etc.);

- Escolher métricas apropriadas (como f1-score, recall, AUC) — e não apenas acurácia.

## Balanceamento das Classes ⚖️ 

Como a base apresenta desbalanceamento de classes (mais clientes que permaneceram do que cancelaram), é importante aplicar técnicas para que os modelos não sejam enviesados em favor da classe majoritária.

### **Objetivos do balanceamento**

- Garantir que os modelos aprendam igualmente sobre clientes que permanecem e que evadem.

- Melhorar métricas como Recall e F1-score, que são mais sensíveis à classe minoritária.

### **Técnicas utilizadas**

Oversampling com SMOTE (Synthetic Minority Over-sampling Technique):

- Gera exemplos sintéticos da classe minoritária a partir dos existentes.

## Análise de Correlação 📈 

A matriz de correlação mostra a relação linear entre variáveis numéricas e Churn (-1 a +1).

- `customer.tenure:` maior correlação absoluta (-0,35). Clientes mais antigos tendem a permanecer.

- `account.Charges.Monthly e account.Charges.Daily:` correlação positiva moderada (~0,19). Cobranças maiores indicam ligeira maior probabilidade de evasão.

- `account.Charges.Total:` correlação negativa fraca (-0,20). Clientes que já pagaram mais tendem a permanecer.

- Alta correlação entre algumas variáveis indica possível redundância, importante considerar na seleção de variáveis.

**Conclusão:** Priorizar customer.tenure, account.Charges.Monthly e account.Charges.Daily na modelagem e reduzir variáveis redundantes.

## Análises Direcionadas 🔍 

### **Violin Plots**

- Tempo de contrato × Churn: clientes com menor tempo de contrato tendem a evadir mais (correlação -0,35).

- Total gasto × Churn: clientes que gastaram mais tendem a permanecer (correlação -0,20).

- Gasto diário × Churn: gastos diários mais altos estão levemente associados à evasão (correlação ~0,19).

### **Scatter Plots**

- Tempo de Contrato vs Total Gasto: clientes que não evadiram concentram-se em tempos e gastos mais altos; os que evadiram aparecem em valores baixos.
- Tempo de Contrato vs Gasto Diário: clientes evadidos estão concentrados nos primeiros meses e apresentam maior dispersão nos gastos diários, enquanto clientes que permaneceram estão distribuídos ao longo do tempo de contrato.

##  Modelagem Preditiva 🤖

### Separação dos Dados
- Conjunto dividido em **70% treino** e **30% teste**.

### Modelos Criados
Foram testados três modelos diferentes para prever a evasão de clientes:

1. **Random Forest** (não exige normalização)  
   - **Acurácia:** 82,5%  
   - Precision, recall e F1-score equilibrados entre as classes  
   - Recall da classe 1 (churn) levemente superior, ideal para identificar clientes propensos a cancelar  
   - **Conclusão:** melhor desempenho geral entre os modelos testados  

2. **KNN** (sensível à escala)  
   - **Antes da padronização:**  
     - Acurácia: 75,7%  
     - Recall para churn (class
3. **Redes Neurais** (necessita normalização)  
   - **Acurácia:** 81%  
   - Precision, recall e F1-score equilibrados para ambas as classes  
   - Baixo número de falsos negativos, essencial para identificar clientes que podem cancelar  
   - **Conclusão:** desempenho sólido, confiável para uso prático em estratégias de retenção

## Avaliação dos Modelos

### Comparação Crítica

- **Melhor desempenho:** O Random Forest obteve a maior acurácia e métricas mais equilibradas, sendo o modelo mais robusto e confiável para este problema.
- **Overfitting/Underfitting:** Nenhum modelo apresenta sinais claros de overfitting (como precisão muito alta e recall baixo) ou underfitting (todas as métricas baixas). O KNN pode estar levemente underfit, pois não captura tão bem as tendências dos dados.
- **Ajustes sugeridos:**
    - Para o KNN: testar diferentes valores de k, normalizar os dados e ajustar hiperparâmetros.
    - Para Redes Neurais: ajustar arquitetura (número de camadas/neuronios), regularização e número de épocas.
    - Para Random Forest: ajustar número de árvores e profundidade, mas o modelo já está com ótimo desempenho.
### Justificativa
O Random Forest é o modelo mais indicado para este caso, pois apresenta melhor equilíbrio entre precisão, recall e f1-score, além de ser menos suscetível a overfitting. Os outros modelos podem ser aprimorados com ajustes, mas atualmente ficam atrás em desempenho.

## Conclusão 🏁

### Principais Fatores Identificados

A análise de importância das variáveis nos três modelos (Random Forest, KNN e Rede Neural) revelou padrões consistentes sobre os fatores que mais influenciam a evasão (churn) de clientes:

1. **Variáveis Financeiras**  
   - `account.Charges.Monthly`, `account.Charges.Total`, `account.Charges.Daily`  
   - Valores cobrados mensalmente, total acumulado e gastos diários têm forte impacto na decisão do cliente de permanecer ou sair. Cobranças mais altas tendem a aumentar a propensão à evasão.

2. **Tempo de Relacionamento**  
   - `customer.tenure`  
   - Clientes mais antigos são mais fiéis, enquanto novos clientes apresentam maior risco de churn.

3. **Tipo e Duração do Contrato**  
   - `account.Contract_Two year`, `account.Contract_One year`  
   - Contratos mais longos estão associados a menor evasão, especialmente nos modelos de Rede Neural.

4. **Serviços de Internet e Suporte**  
   - `internet.TechSupport`, `internet.OnlineSecurity`, `internet.InternetService_Fiber optic`, `internet.InternetService_No`  
   - Presença de suporte técnico, segurança online e tipo de serviço de internet impactam a satisfação do cliente. Ausência ou falhas nesses serviços aumentam o risco de churn.

5. **Fatores Demográficos e de Pagamento**  
   - `customer.Dependents`, `customer.Partner`, `customer.SeniorCitizen`, `account.PaymentMethod_CreditCard`, `account.PaymentMethod_ElectronicCheck`, `account.PaperlessBilling`  
   - Embora menos relevantes, contribuem para o modelo. Clientes com dependentes ou parceiros tendem a ser mais estáveis, enquanto métodos de pagamento e idade têm influência secundária.


## Certificação 🥇

<img width="500" height="500" alt="badge telexom x 2" src="https://github.com/user-attachments/assets/cc16389e-180c-4db8-a826-98c124b65689" />

 
