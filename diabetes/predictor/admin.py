from django.contrib import admin
from .models import Patient, Dataset

admin.site.register(Patient)
admin.site.register(Dataset)
