"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path

from accounts.views import TokenRevokingPasswordChangeView
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path

from accounts.views import TokenRevokingPasswordChangeView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('meals.urls')),
    # Django's built-in auth views: login, logout, password change/reset.
    # Templates live in templates/registration/.
    # Registered BEFORE the include so it wins resolution for this path:
    # password change must also revoke the user's API token (see accounts.views).
    path('accounts/password_change/',
         TokenRevokingPasswordChangeView.as_view(), name='password_change'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('accounts/', include('accounts.urls')),
    path('api/', include('api.urls')),
]
