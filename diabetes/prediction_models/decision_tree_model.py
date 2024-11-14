import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Загрузка данных
csv_path = 'data/diabetes_1.csv'
data = pd.read_csv(csv_path)

# Замена NaN значений на нули
#data = data.fillna(0)

# Разделение данных на признаки и целевую переменную
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Обучение модели дерева решений
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Сохранение модели
joblib.dump(decision_tree, 'decision_tree_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Функция для предсказания
def predict_diabetes(input_data):
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('decision_tree_model.joblib')
    input_data_scaled = scaler.transform(input_data)
    prediction_proba = model.predict_proba(input_data_scaled)[0][1]
    return prediction_proba
