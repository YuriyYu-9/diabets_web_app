from django.contrib.auth.models import User
from django.db import models
from django.conf import settings

class Patient(models.Model):
    # Ссылка на пользователя, создавшего запись пациента
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    pregnancies = models.IntegerField(help_text="Number of pregnancies")
    glucose = models.IntegerField(help_text="Glucose level")
    blood_pressure = models.IntegerField(help_text="Blood pressure")
    skin_thickness = models.IntegerField(help_text="Skin thickness")
    insulin = models.IntegerField(help_text="Insulin level")
    bmi = models.FloatField(help_text="Body Mass Index")
    diabetes_pedigree_function = models.FloatField(help_text="Diabetes pedigree function")
    age = models.IntegerField(help_text="Age of the patient")
    outcome = models.BooleanField(default=False, help_text="Diabetes prediction outcome")

    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.outcome}"

class Dataset(models.Model):
    pregnancies = models.IntegerField(help_text="Number of pregnancies")
    glucose = models.IntegerField(help_text="Glucose level")
    blood_pressure = models.IntegerField(help_text="Blood pressure")
    skin_thickness = models.IntegerField(help_text="Skin thickness")
    insulin = models.IntegerField(help_text="Insulin level")
    bmi = models.FloatField(help_text="Body Mass Index")
    diabetes_pedigree_function = models.FloatField(help_text="Diabetes pedigree function")
    age = models.IntegerField(help_text="Age")
    outcome = models.BooleanField(default=False, help_text="Diagnosed diabetes")

    def __str__(self):
        return f"Record {self.id}: Diabetes - {'Yes' if self.outcome else 'No'}"
