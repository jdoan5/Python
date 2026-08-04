"""API URLconf — the router generates the CRUD routes from the ViewSet.

    /api/meals/           GET list, POST create
    /api/meals/<pk>/      GET, PUT/PATCH, DELETE
    /api/meals/summary/   GET (the @action)
    /api/token/           POST username+password -> {"token": "..."}
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import MealViewSet, ThrottledObtainAuthToken

router = DefaultRouter()
router.register("meals", MealViewSet, basename="meal")

app_name = "api"

urlpatterns = [
    path("token/", ThrottledObtainAuthToken.as_view(), name="token"),
    path("", include(router.urls)),
]
