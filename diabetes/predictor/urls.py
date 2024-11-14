from django.urls import path
from .views import login_view, index_view, predictions_view, eda_view, eda_data, eda_plot, eda_histograms, eda_boxplots, eda_pairplots, logout_view, patients_view, add_patient, update_patient, delete_patient, edit_patient
#logout_view

urlpatterns = [
    path('index/', index_view, name='index'),
    path('prediction/', predictions_view, name='prediction'),  # Проверка маршрута
    path('login/', login_view, name='login'),
    path('eda', eda_view, name='eda'),
    path('eda_data', eda_data, name='eda_data'),
    path('eda/plot/', eda_plot, name='eda_plot'),
    path('eda/eda_histograms/', eda_histograms, name='eda_histograms'),
    path('eda/eda_boxplots/', eda_boxplots, name='eda_boxplots'),
    path('eda/eda_pairplots/', eda_pairplots, name='eda_pairplots'),
    path('logout/', logout_view, name='logout'),  # Добавляем путь для logout
    path('patients/', patients_view, name='patients'),
    path('patients/add/', add_patient, name='add_patient'),
    path('patients/<int:pk>/update/', update_patient, name='update_patient'),
    path('patients/<int:pk>/delete/', delete_patient, name='delete_patient'),
    path('patients/<int:pk>/edit/', edit_patient, name='edit_patient'),
        #path('logout/', logout_view, name='logout'),
]
