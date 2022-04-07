## Bibliotecas necessárias
import pandas as pd
import seaborn as sn

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


## Leitura do Arquivo
dataframe = pd.read_csv('https://raw.githubusercontent.com/REXDES/N1-IA-Projeto/main/2%20Year%20IBM%20Stock%20Data.csv')
dataframe = dataframe.dropna()      # Tira os dados nulos

# Adicionando coluna superavit, que classifica binariamente se uma ação 
# subiu ou não, em relação ao preço de abertura e fechamento do dia.
dataframe['superavit'] = (dataframe['close'] > dataframe['open'])/1
dataframe.head()


## Regressão Linear

# Criando os dados
x = dataframe.drop(columns=['superavit'])
y = dataframe['superavit']

# Treinamento
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=1984)

# Base de teste para ser carregada no Modelo
new_dataframe = pd.concat([X_train, y_train], axis=1)

# Modelo
model = sm.ols(formula="superavit ~ open + close", data=new_dataframe)

# Resultados
result = model.fit()
print(result.summary())


y_pred = result.predict(X_test)

# Retornando os dados float para binariedade
y_pred = (y_pred > 0.5)/1

acuracidade = sum(y_pred == y_test)*100 / len(y_pred)
print(f"\n\nAcuracidade: {acuracidade:.3f}.")




### Regressão Logísitca
df_filtrado = dataframe[['superavit', 'open', 'close']]

x = df_filtrado.drop(columns=['superavit'])
y = df_filtrado['superavit']

#separando modelo para treinamento/teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=1984)

reg_logistica = LogisticRegression()
reg_logistica.fit(X_train, y_train)


y_pred = reg_logistica.predict(X_test)

acuracidade = sum(y_pred == y_test)*100 / len(y_pred)
print(f"Acuracidade: {acuracidade:.3f}.")



confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
