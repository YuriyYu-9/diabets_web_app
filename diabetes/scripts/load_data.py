import csv
import os
import sys
import django

# Задаем настройки Django
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'diabetes.settings')
django.setup()

from predictor.models import Dataset

# Путь к CSV-файлу (два уровня вверх от текущей директории)
csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'diabetes.csv')

# Функция для загрузки данных из CSV в базу данных
def load_data_from_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = Dataset(
                glucose=row['Glucose'],
                blood_pressure=row['BloodPressure'],
                skin_thickness=row['SkinThickness'],
                insulin=row['Insulin'],
                bmi=row['BMI'],
                diabetes_pedigree_function=row['DiabetesPedigreeFunction'],
                age=row['Age'],
                outcome=row['Outcome'] == '1'
            )
            dataset.save()

# Выполняем загрузку данных
load_data_from_csv(csv_file_path)
