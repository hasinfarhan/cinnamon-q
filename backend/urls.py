"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from api.views import hello_world
from django.http import JsonResponse

# Define a simple home view
def home_view(request):
    return JsonResponse({"message": "Welcome to the API"})

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view),  # Add this line to serve the home route
    path('api/hello/', hello_world),
]