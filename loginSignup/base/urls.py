
from django.urls import path, include
from .views import authView, home, custom_logout


urlpatterns = [
    path("", home, name="home"),
    path("signup/", authView, name="authView"), 
    path("accounts/", include('django.contrib.auth.urls')),
    path("logout/", custom_logout, name="custom_logout"),
    
]