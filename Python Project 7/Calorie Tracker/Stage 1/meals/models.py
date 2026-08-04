"""The Meal model — one row per logged meal.

Django learning notes:
- Field types map to DB columns; validators run in forms and full_clean(),
  NOT automatically on .save() — that's why the form layer matters.
- `Meta.ordering` gives every queryset a default ORDER BY.
- `eaten_at` is timezone-aware (USE_TZ=True): stored as UTC in the DB,
  displayed and date-filtered in the project TIME_ZONE.
"""

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


class Meal(models.Model):
    class MealType(models.TextChoices):
        BREAKFAST = "breakfast", "Breakfast"
        LUNCH = "lunch", "Lunch"
        DINNER = "dinner", "Dinner"
        SNACK = "snack", "Snack"

    name = models.CharField(max_length=120)
    meal_type = models.CharField(
        max_length=12, choices=MealType.choices, default=MealType.SNACK
    )
    calories = models.PositiveIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5000)],
        help_text="Calories for this meal (1-5000).",
    )
    eaten_at = models.DateTimeField(default=timezone.now)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-eaten_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.calories} kcal)"
