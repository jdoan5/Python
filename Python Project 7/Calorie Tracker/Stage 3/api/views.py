"""API views — one ModelViewSet gives the full CRUD surface.

DRF learning notes:
- ModelViewSet + a router = list/retrieve/create/update/delete with correct
  status codes, from ~15 lines.
- get_queryset() filtering by request.user is the SAME scope rule as the
  Stage 2 pages — and it makes cross-user access a 404 (the object simply
  isn't in your queryset), which leaks less than a 403 would.
- perform_create() is the hook where server-side facts (ownership) are
  attached to client data.
- @action adds custom endpoints to the ViewSet's router registration:
  summary -> /api/meals/summary/.
"""

from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from rest_framework import viewsets
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle

from meals.models import Meal

from .serializers import DailySummarySerializer, MealSerializer


class ThrottledObtainAuthToken(ObtainAuthToken):
    """Token endpoint with a brute-force cap.

    DRF deliberately ships ObtainAuthToken with `throttle_classes = ()` —
    an empty tuple that OVERRIDES any DEFAULT_THROTTLE_CLASSES — leaving
    rate-limiting of password guesses to you. Without this, /api/token/ is
    an unmetered password oracle. Rate lives in settings:
    REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]["token_obtain"].
    """

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "token_obtain"


class MealViewSet(viewsets.ModelViewSet):
    serializer_class = MealSerializer

    def get_queryset(self):
        # The scope rule, API edition. Meal.objects.all() here would be the
        # data leak the Stage 2 README warns about.
        return Meal.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False)
    def summary(self, request):
        """Daily totals (last 30 days with data), oldest first — chart-ready."""
        rows = list(
            self.get_queryset()
            .annotate(day=TruncDate("eaten_at"))
            .values("day")
            .annotate(total=Sum("calories"), count=Count("id"))
            .order_by("-day")[:30]
        )[::-1]
        return Response(DailySummarySerializer(rows, many=True).data)
