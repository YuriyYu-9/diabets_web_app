from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from .forms import LoginForm, PredictionForm, PatientForm
from .models import Patient, Dataset
from prediction_models.logreg_model import predict_diabetes as logreg_predict
from prediction_models.decision_tree_model import predict_diabetes as dt_predict
from prediction_models.cnn_model import predict_diabetes as cnn_predict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.conf import settings
import os
import urllib.parse
import numpy as np
from sklearn.linear_model import LinearRegression
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .forms import PatientForm
from .models import Patient

def redirect_to_login(request):
    return redirect('login')

def logout_view(request):
    logout(request)
    return redirect('login')

def login_view(request):
    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('index')
        return render(request, 'predictor/login.html', {'form': form})
    else:
        form = LoginForm()
    return render(request, 'predictor/login.html', {'form': form})

@login_required
def index_view(request):
    csv_path = os.path.join(settings.BASE_DIR, 'data/diabetes.csv')
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        return render(request, 'predictor/error.html', {'message': 'Файл не найден. Пожалуйста, убедитесь, что файл существует.'})

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for i, column in enumerate(columns):
        sns.histplot(data[column], ax=axes[i // 3, i % 3])
        axes[i // 3, i % 3].set_title(f'Distribution of {column}')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_png = buf.getvalue()
    buf.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return render(request, 'predictor/index.html', {'graphic': graphic})

def create_probability_chart(probability):
    fig, ax = plt.subplots()
    bars = ax.bar(["No Diabetes", "Diabetes"], [1 - probability, probability], color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Diabetes Prediction Probability')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')  # добавление текста на график

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return uri

@login_required
def patients_view(request):
    patients = Patient.objects.filter(created_by=request.user)
    form = PatientForm()
    return render(request, 'predictor/patients.html', {'patients': patients, 'form': form})

@login_required
def add_patient(request):
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save(commit=False)
            patient.created_by = request.user
            patient.save()
            return JsonResponse({'message': 'Data Base Update Success'})
        return JsonResponse({'error': 'Invalid data'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def update_patient(request, pk):
    patient = get_object_or_404(Patient, pk=pk)
    if request.method == 'POST':
        form = PatientForm(request.POST, instance=patient)
        if form.is_valid():
            form.save()
            return JsonResponse({'message': 'Data Base Update Success'})
        return JsonResponse({'error': 'Invalid data'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def delete_patient(request, pk):
    patient = get_object_or_404(Patient, pk=pk)
    if request.method == 'POST':
        patient.delete()
        return JsonResponse({'message': 'Data Base Update Success'})
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def edit_patient(request, pk):
    patient = get_object_or_404(Patient, pk=pk)
    form = PatientForm(instance=patient)
    return JsonResponse({'form': form.as_p()})

@login_required
def eda_view(request):
    csv_path = os.path.join(settings.BASE_DIR, 'data/diabetes_1.csv')
    data = pd.read_csv(csv_path)

    # Создание корреляционной матрицы
    correlation_matrix = data.corr().to_html(classes='table table-striped table-hover')

    # Отображение первых 10 строк по умолчанию
    data_html = data.head(10).to_html(classes='table table-striped table-hover')

    return render(request, 'predictor/eda.html', {
        'data': data_html,
        'correlation_matrix': correlation_matrix
    })

@login_required
def eda_data(request):
    rows = request.GET.get('rows', '10')
    csv_path = os.path.join(settings.BASE_DIR, 'data/diabetes_1.csv')
    data = pd.read_csv(csv_path)

    if rows == 'all':
        data_html = data.to_html(classes='table table-striped table-hover')
    else:
        rows = int(rows)
        data_html = data.head(rows).to_html(classes='table table-striped table-hover')

    return JsonResponse({'data': data_html})

@login_required
def eda_plot(request):
    param1 = request.GET.get('param1', 'Glucose')
    param2 = request.GET.get('param2', 'BMI')
    show_regression = request.GET.get('regression_line', 'false') == 'true'

    csv_path = os.path.join(settings.BASE_DIR, 'data/diabetes_1.csv')
    data = pd.read_csv(csv_path)

    x = data[param1].tolist()
    y = data[param2].tolist()

    response = {'x': x, 'y': y}

    if show_regression:
        x_array = np.array(x).reshape(-1, 1)
        y_array = np.array(y)
        model = LinearRegression()
        model.fit(x_array, y_array)
        y_pred = model.predict(x_array)
        response['regression_x'] = x
        response['regression_y'] = y_pred.tolist()

    return JsonResponse(response)

@login_required
def eda_histograms(request):
    param = request.GET.get('param', None)
    csv_path = os.path.join(settings.BASE_DIR, 'data/diabetes_1.csv')
    data = pd.read_csv(csv_path)

    histograms = []
    if param:
        histograms.append({
            'x': data[param].tolist(),
            'name': param
        })
    else:
        columns = data.columns
        for column in columns:
            histograms.append({
                'x': data[column].tolist(),
                'name': column
            })

    return JsonResponse({'histograms': histograms})

@login_required
def eda_boxplots(request):
    param = request.GET.get('param', None)
    csv_path = os.path.join(settings.BASE_DIR, 'data/diabetes_1.csv')
    data = pd.read_csv(csv_path)

    boxplots = []
    if param:
        boxplots.append({
            'y': data[param].tolist(),
            'name': param
        })
    else:
        columns = data.columns
        for column in columns:
            boxplots.append({
                'y': data[column].tolist(),
                'name': column
            })

    return JsonResponse({'boxplots': boxplots})

@login_required
def eda_pairplots(request):
        param1 = request.GET.get('param1', None)
        param2 = request.GET.get('param2', None)

        csv_path = os.path.join(settings.BASE_DIR, 'data/diabetes_1.csv')
        data = pd.read_csv(csv_path)

        pairplots = []
        if param1 and param2:
            pairplots.append({
                'x': data[param1].tolist(),
                'y': data[param2].tolist(),
                'name': f'{param1} vs {param2}'
            })
        else:
            columns = data.columns
            for i, column1 in enumerate(columns):
                for column2 in columns[i + 1:]:
                    pairplots.append({
                        'x': data[column1].tolist(),
                        'y': data[column2].tolist(),
                        'name': f'{column1} vs {column2}'
                    })

        return JsonResponse({'pairplots': pairplots})



@login_required
def predictions_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']
            pregnancies = form.cleaned_data['pregnancies']
            glucose = form.cleaned_data['glucose']
            blood_pressure = form.cleaned_data['blood_pressure']
            skin_thickness = form.cleaned_data['skin_thickness']
            insulin = form.cleaned_data['insulin']
            bmi = form.cleaned_data['bmi']
            diabetes_pedigree_function = form.cleaned_data['diabetes_pedigree_function']
            age = form.cleaned_data['age']
            algorithm = form.cleaned_data['algorithm']

            patient, created = Patient.objects.get_or_create(
                first_name=first_name,
                last_name=last_name,
                created_by=request.user,
                defaults={
                    'pregnancies': pregnancies,
                    'glucose': glucose,
                    'blood_pressure': blood_pressure,
                    'skin_thickness': skin_thickness,
                    'insulin': insulin,
                    'bmi': bmi,
                    'diabetes_pedigree_function': diabetes_pedigree_function,
                    'age': age
                }
            )

            if not created:
                patient.pregnancies = pregnancies
                patient.glucose = glucose
                patient.blood_pressure = blood_pressure
                patient.skin_thickness = skin_thickness
                patient.insulin = insulin
                patient.bmi = bmi
                patient.diabetes_pedigree_function = diabetes_pedigree_function
                patient.age = age
                patient.save()

            input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]

            if algorithm == 'cnn':
                prediction_proba = cnn_predict(input_data)
            elif algorithm == 'logreg':
                prediction_proba = logreg_predict(input_data)
            elif algorithm == 'decision_tree':
                prediction_proba = dt_predict(input_data)
            else:
                prediction_proba = 0.5  # default value if no valid algorithm is selected

            is_diabetic = prediction_proba >= 0.5
            outcome = int(is_diabetic)

            # Сохранение данных в базу данных
            Dataset.objects.create(
                pregnancies=pregnancies,
                glucose=glucose,
                blood_pressure=blood_pressure,
                skin_thickness=skin_thickness,
                insulin=insulin,
                bmi=bmi,
                diabetes_pedigree_function=diabetes_pedigree_function,
                age=age,
                outcome=outcome
            )

            # Обновление CSV-файла
            csv_file_path = os.path.join(settings.BASE_DIR, 'data', 'diabetes_1.csv')
            if os.path.exists(csv_file_path):
                data = pd.read_csv(csv_file_path)
            else:
                data = pd.DataFrame(columns=[
                    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age', 'Outcome'
                ])

            new_row = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree_function,
                'Age': age,
                'Outcome': outcome
            }

            new_data = pd.DataFrame([new_row])
            data = pd.concat([data, new_data], ignore_index=True)
            data.fillna(0, inplace=True)
            data.to_csv(csv_file_path, index=False)

            chart = create_probability_chart(prediction_proba)

            return render(request, 'predictor/prediction_result.html', {
                'patient': patient,
                'is_diabetic': is_diabetic,
                'chart': chart
            })
    else:
        form = PredictionForm()

    return render(request, 'predictor/predictions.html', {'form': form})
