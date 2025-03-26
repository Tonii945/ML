from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Preprocesamiento 
df = pd.read_csv('data/raw/bank-marketing-campaign-data.csv', sep=';')
df.head()
df.info()
df.describe(include='all')
sns.countplot(data=df, x='y')
plt.show()

sns.histplot(df['age'], bins=30)
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(y='job', hue='y', data=df)
plt.show()

categoricas = df.select_dtypes(include=['object']).columns.drop('y')
df_procesado = pd.get_dummies(df, columns=categoricas, drop_first=True)
df_procesado['y'] = df_procesado['y'].map({'yes':1, 'no':0})

corr = df_procesado.corr()['y'].abs().sort_values(ascending=False)
print(corr)

baja_corr = corr[corr < 0.05].index
df_procesado.drop(columns=baja_corr, inplace=True)

for col in df_procesado.columns:
    if df_procesado[col].value_counts(normalize=True).max() > 0.95:
         df_procesado.drop(columns=col, inplace=True)
# División train-test

X = df_procesado.drop('y', axis=1)
y = df_procesado['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pd.DataFrame(X_train).to_csv('data/processed/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

#Regresión logística simple
modelo = LogisticRegression(class_weight='balanced', max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()

# Modelo optimizado

parametros = {'C': [0.01, 0.1, 1, 10, 100]}

busqueda = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000),
    parametros,
    cv=5,
    scoring='accuracy'
)

busqueda.fit(X_train, y_train)

print("Mejores parámetros:", busqueda.best_params_)
print("Accuracy optimizado:", busqueda.best_score_)

modelo_optimizado = busqueda.best_estimator_
y_pred_optimizado = modelo_optimizado.predict(X_test)


# Resultados optimizado
print("Accuracy final optimizado:", accuracy_score(y_test, y_pred_optimizado))
print(classification_report(y_test, y_pred_optimizado))

