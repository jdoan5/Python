"""Views — where HTTP meets the ORM.

Deliberately mixed styles so both patterns are learned:
- `today` and `day_detail` are function-based views (FBVs): explicit and
  readable, the best way to see the request/response cycle.
- `HistoryView` is a class-based ListView: less code once you know the
  conventions; you meet get_queryset() and context_object_name.

Patterns worth internalizing here:
- POST → validate → save → REDIRECT (the Post/Redirect/Get pattern) so a
  browser refresh never double-submits the form.
- Aggregation happens in the database (Sum, Count, TruncDate), not in
  Python — the ORM builds GROUP BY queries for you.
"""

from django.conf import settings
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.generic import ListView

from .forms import MealForm
from .models import Meal


def today(request):
    """Log a meal + see today's running total against the daily goal."""
    if request.method == "POST":
        form = MealForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("meals:today")  # PRG: refresh-safe
    else:
        form = MealForm()

    local_today = timezone.localdate()
    meals = Meal.objects.filter(eaten_at__date=local_today)
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


class HistoryView(ListView):
    """Daily totals, newest first — one GROUP BY query."""

    template_name = "meals/history.html"
    context_object_name = "days"
    paginate_by = 14

    def get_queryset(self):
        return (
            Meal.objects.annotate(day=TruncDate("eaten_at"))
            .values("day")
            .annotate(total=Sum("calories"), count=Count("id"))
            .order_by("-day")
        )


def day_detail(request, day):
    """All meals for one date (URL: /history/2026-08-04/).

    `day` arrives as a datetime.date — the isodate path converter in urls.py
    parsed and validated it; malformed dates 404 before reaching this view.
    """
    meals = Meal.objects.filter(eaten_at__date=day)
    total = meals.aggregate(total=Sum("calories"))["total"] or 0
    return render(
        request,
        "meals/day_detail.html",
        {"day": day, "meals": meals, "total": total, "goal": settings.DAILY_CALORIE_GOAL},
    )
