# Health-Insurance-Cross-Sell
Identificação de Clientes Propensos a Adquirir Seguro Residencial


PROBLEMA:
Uma seguradora líder no ramo de automóveis, visando expandir seu portfólio, decidiu explorar o potencial de oferecer um novo produto: seguro residencial. Para conduzir essa iniciativa estratégica, a empresa realizou uma abrangente pesquisa com 380 mil clientes, tanto atuais quanto antigos, buscando entender as necessidades e preferências desse vasto público.

Nessa extensa análise, coletamos informações valiosas sobre os hábitos residenciais, históricos de sinistros e preferências de cobertura dos clientes. Utilizamos variáveis como:

id: Identificação única do cliente.
Gender: Gênero do cliente.
Age: Idade do cliente.
Driving_License: Indicação se o cliente possui carteira de motorista.
Region_Code: Código da região do cliente.
Previously_Insured: Indicação se o cliente tinha seguro anteriormente.
Vehicle_Age: Idade do veículo do cliente.
Vehicle_Damage: Indicação se o veículo do cliente teve danos anteriores.
Annual_Premium: Valor do prêmio anual do seguro.
Policy_Sales_Channel: Canal de venda do seguro.
Vintage: Tempo em dias desde a primeira interação do cliente com a seguradora.
Response: Indicação se o cliente mostrou interesse no novo produto de seguro residencial.

Diante desse cenário, o time de dados foi convocado para uma nova missão crucial: desenvolver um modelo de machine learning para identificar os clientes com maior propensão a aderir ao novo seguro residencial. 

Dada a capacidade limitada do time de vendas, o cientista de dados criou um modelo integrado ao Google Sheets, proporcionando o acesso à equipe de vendas a um ranking dinâmico dos clientes mais propensos, permitindo uma abordagem focada e eficiente. Essa otimização estratégica direcionou os esforços da equipe para aqueles mais propensos a aderir ao novo seguro residencial, maximizando o impacto das interações comerciais. 


OBJETIVO:
Auxiliar o time de vendas a focar os esforços na captação de novos clientes para o seguro residencial. Através de um ranqueamento de clientes mais propensos a contratar o novo produto. Meta a ser atingida: Dobrar a quantidade de clientes adiquirindo o novo produto, em comparação com uma prospecção de clientes realizada por uma amostra aleatória, ao ligar para 20% da base de clientes.

Seleção de Variáveis:
   - Na etapa de Feature Selection, aplicamos o ExtraTreesClassifier para identificar as features mais relevantes. Abaixo está a lista das features e suas "taxas de importância" calculadas pelo ExtraTreesClassifier:

------------------------------ Mostrar print ------------------------------

Algoritmo de Machine Learning:
   - O algoritmo escolhido foi o ExtraTreesClassifier. Realizamos testes com 4 algoritmos diferentes (KNN, Logistic Regression, Extra Trees Classifier e Random Forest) e selecionamos o melhor modelo com base em gráficos de Cumulative Gain e Lift Curve.

------------------------------ Mostrar print ------------------------------

Abordagem de Treinamento do Modelo:
   - Os dados foram divididos entre treino e teste na razão de 80% para treino e 20% para teste.
   - Não foram aplicadas técnicas específicas para lidar com o desequilíbrio de classe.

### Análise Exploratória de Dados (EDA):

#### Idade (Age):
- O boxplot revelou uma mediana próxima dos 34 anos para os clientes que responderam positivamente à pesquisa, enquanto para os demais, a mediana marcou próximo de 35 anos.
- O histograma destacou uma grande concentração dos dados próximo aos 45 anos para os clientes que responderam positivamente à pesquisa, enquanto teve uma maior concentração aos 25 anos para os demais casos.

#### Prêmio Anual (Annual Premium):
- O boxplot identificou uma grande quantidade de outliers para ambos os grupos (respostas sim e não à pesquisa). O 4º quartil ficou abaixo de 100,000, com outliers variando até 500,000. A mediana de ambos os casos ficou próxima de 30,000.
- Histogramas revelaram semelhanças marcantes entre ambos os casos, destacando uma alta concentração em um valor específico (2630), que provavelmente representa um valor mínimo. Após remover os outliers, ambas as distribuições se aproximaram de uma normal.

#### Carteira de Motorista (Driving License):
- A presença ou ausência da carteira de motorista afeta significativamente o interesse. Clientes sem carteira de motorista mostraram uma redução de aproximadamente metade nas respostas positivas.

#### Código da Região (Region Code):
- Identificou-se que a região 28 se destaca nas respostas positivas e negativas. A região 28 é a que possui a maior quantidade de clientes com interesse, confirmada também como a mesma região com a maior quantidade de clientes sem interesse.

#### Seguro Anterior (Previously Insured):
- Uma diferença significativa foi observada na proporção de clientes interessados e não interessados em três cenários: clientes sem seguro anterior, clientes com seguro anterior e todos os clientes.
- Clientes sem seguro anterior mostraram menos interesse (0.09%), significativamente inferior aos 12% do total da pesquisa.
- Clientes com seguro anterior mostraram mais interesse (22.55%), quase o dobro da média total.

#### Canal de Venda do Seguro (Policy Sales Channel):
- Dois casos se destacaram com os números 26 e 124.

#### Tempo Desde a Primeira Interação (Vintage):
- Não foi identificada variação significativa entre clientes interessados e não interessados.

Matriz de correlação:

------------------------------ Mostrar print ------------------------------

Veículo Danificado (Vehicle Damage): Correlação positiva significativa com o interesse no novo seguro. Clientes com veículos danificados têm maior probabilidade de expressar interesse.
Seguro Anterior (Previously Insured): Correlação negativa notável. Clientes que já possuíam seguro anteriormente têm menor propensão a manifestar interesse no novo seguro residencial.
Idade (Age): Correlação positiva leve. Indica que clientes ligeiramente mais jovens podem ter uma tendência sutilmente maior de demonstrar interesse.


### Preparação dos Dados:

Para garantir a eficácia do modelo de machine learning, foram realizadas as seguintes etapas de preparação dos dados:

#### Padronização e Normalização:
- A feature 'annual_premium' foi padronizada usando StandardScaler, devida a quantidade de outliers.
- As features 'age' e 'vintage' foram normalizadas utilizando MinMaxScaler.

#### Codificação de Variáveis Categóricas:
- As features 'gender' e 'region_code' foram codificadas usando Target Encoding. Substituindo cada categoria por sua média de resposta (proporção de clientes interessados). Isso ajuda o modelo a aprender a relação entre essas categorias e a variável resposta, considerando a taxa de interesse média para cada valor categórico.
- A feature 'policy_sales_channel' foi codificada usando Target Encoding , visando modelar a relação entre os canais de venda e a variável resposta, considerando as taxas de resposta médias para cada canal.
- A feature 'vehicle_age' foi expandida usando One Hot Encoding.

Essas etapas asseguram a consistência e relevância dos dados para o treinamento do modelo, contribuindo para uma previsão mais precisa do interesse dos clientes no novo seguro residencial.


