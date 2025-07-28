from django.urls import path
from App import views

urlpatterns = [
    path('',views.form_load,name="form_load"),
    path('generate_response/',views.generate_response,name="generate_response"),
    
]