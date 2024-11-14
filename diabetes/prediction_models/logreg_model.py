import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# Загрузка данных
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes_1.csv')
data = pd.read_csv(csv_path)

# Замена NaN значений на нули
#data = data.fillna(0)

# Определение признаков и целевой переменной
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание и обучение модели
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

def predict_diabetes(input_data):
    input_data = scaler.transform(input_data)
    return logreg.predict_proba(input_data)[0][1]
