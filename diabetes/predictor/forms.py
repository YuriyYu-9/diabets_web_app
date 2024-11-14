from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .models import Patient

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'input',
        'placeholder': 'Username'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'input',
        'placeholder': 'Password'
    }))

class PredictionForm(forms.ModelForm):
    ALGORITHM_CHOICES = [
        ('cnn', 'CNN'),
        ('logreg', 'Logistic Regression'),
        ('decision_tree', 'Decision Tree')
    ]

    algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES, widget=forms.Select(attrs={'class': 'input'}))

    class Meta:
        model = Patient
        fields = ['first_name', 'last_name', 'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'input', 'placeholder': 'First Name'}),
            'last_name': forms.TextInput(attrs={'class': 'input', 'placeholder': 'Last Name'}),
            'pregnancies': forms.NumberInput(attrs={'class': 'input', 'placeholder': 'Number of Pregnancies'}),
            'glucose': forms.NumberInput(attrs={'class': 'input', 'placeholder': 'Glucose Level'}),
            'blood_pressure': forms.NumberInput(attrs={'class': 'input', 'placeholder': 'Blood Pressure'}),
            'skin_thickness': forms.NumberInput(attrs={'class': 'input', 'placeholder': 'Skin Thickness'}),
            'insulin': forms.NumberInput(attrs={'class': 'input', 'placeholder': 'Insulin Level'}),
            'bmi': forms.NumberInput(attrs={'class': 'input', 'placeholder': 'BMI'}),
            'diabetes_pedigree_function': forms.NumberInput(attrs={'class': 'input', 'placeholder': 'Diabetes Pedigree Function'}),
            'age': forms.NumberInput(attrs={'class': 'input', 'placeholder': 'Age'}),
        }
class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['first_name', 'last_name', 'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
