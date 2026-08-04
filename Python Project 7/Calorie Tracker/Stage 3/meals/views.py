"""Views — Stage 2 adds the two auth patterns every Django app needs:

1. GATE: @login_required (FBVs) / LoginRequiredMixin (CBVs) redirect
   anonymous visitors to settings.LOGIN_URL with ?next= for the round-trip.
2. SCOPE: every queryset filters user=request.user, and saves stamp the
   user via form.save(commit=False). Forgetting the filter on even ONE
   query is how "I can see someone else's data" bugs ship — the tests
   assert isolation on every view.

Stage 1 patterns still on display: FBV vs CBV, Post/Redirect/Get, DB-side
aggregation (Sum/Count/TruncDate).
"""

import csv

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.generic import ListView

from .forms import MealForm
from .models import Meal


@login_required
def today(request):
    """Log a meal + see today's running total against the daily goal."""
    if request.method == "POST":
        form = MealForm(request.POST)
        if form.is_valid():
            meal = form.save(commit=False)  # don't hit the DB yet...
            meal.user = request.user        # ...stamp ownership first
            meal.save()
            return redirect("meals:today")  # PRG: refresh-safe
    else:
        form = MealForm()

    local_today = timezone.localdate()
    meals = Meal.objects.filter(user=request.user, eaten_at__date=local_today)
    total = meals.aggregate(total=Sum("calories"))["total"] or 0
    goal = settings.DAILY_CALORIE_GOAL
    percent = min(100, round(total * 100 / goal)) if goal else 0

    return render(
        request,
        "meals/today.html",
        {
            "form": form,
            "meals": meals,
            "total": total,
            "goal": goal,
            "percent": percent,
            "over_goal": total > goal,
            "over_by": max(0, total - goal),
            "today": local_today,
        },
    )


class HistoryView(LoginRequiredMixin, ListView):
    """Daily totals for the signed-in user, newest first — one GROUP BY."""

    template_name = "meals/history.html"
    context_object_name = "days"
    paginate_by = 14

    def get_queryset(self):
        return (
            Meal.objects.filter(user=self.request.user)
            .annotate(day=TruncDate("eaten_at"))
            .values("day")
            .annotate(total=Sum("calories"), count=Count("id"))
            .order_by("-day")
        )


@login_required
def day_detail(request, day):
    """The signed-in user's meals for one date (URL: /history/2026-08-04/)."""
    meals = Meal.objects.filter(user=request.user, eaten_at__date=day)
    total = meals.aggregate(total=Sum("calories"))["total"] or 0
    return render(
        request,
        "meals/day_detail.html",
        {"day": day, "meals": meals, "total": total, "goal": settings.DAILY_CALORIE_GOAL},
    )


@login_required
def export_csv(request):
    """Download the signed-in user's full meal log as CSV.

    No DRF here on purpose: a file download is a plain Django view — set the
    content type, add a Content-Disposition header, and hand the response
    object to csv.writer as if it were a file.
    """
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="meals.csv"'
    response.write("\ufeff")  # UTF-8 BOM: Excel needs it to detect the encoding
    writer = csv.writer(response)
    writer.writerow(["eaten_at", "name", "meal_type", "calories", "notes"])
    meals = Meal.objects.filter(user=request.user).order_by("eaten_at")
    for meal in meals:
        writer.writerow([
            timezone.localtime(meal.eaten_at).strftime("%Y-%m-%d %H:%M"),
            meal.name,
            meal.meal_type,
            meal.calories,
            meal.notes,
        ])
    return response
