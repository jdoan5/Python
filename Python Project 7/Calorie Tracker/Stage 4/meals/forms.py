"""MealForm — a ModelForm, Django's biggest time-saver.

A ModelForm derives its fields, labels, and base validation from the model.
You add anything the model can't express — here, "eaten_at can't be in the
future", which needs the current time and so can't live on the model field.

clean_<fieldname>() methods are the per-field validation hook; raise
ValidationError and the form re-renders with the message next to the field.

Timezone subtlety (found by review, verified empirically): with USE_TZ,
Django's DateTimeField REJECTS wall-clock times in the DST fall-back hour
(1:00-1:59 AM repeats on the November transition) as "ambiguous" — meaning a
user literally could not log a 1:30 AM snack that night. FoldTolerantDateTimeField
resolves ambiguity to the first occurrence (fold=0) instead of erroring.
"""

from datetime import datetime

from django import forms
from django.core.exceptions import ValidationError
from django.utils import timezone

from .models import Meal


class FoldTolerantDateTimeField(forms.DateTimeField):
    """Accept DST-ambiguous wall times instead of rejecting them.

    Ambiguous times resolve to their first occurrence (fold=0, i.e. still on
    daylight time). Nonexistent spring-forward times normalize forward, which
    zoneinfo does automatically.
    """

    def to_python(self, value):
        try:
            return super().to_python(value)
        except ValidationError:
            try:
                naive = datetime.fromisoformat(str(value))
            except (TypeError, ValueError):
                raise  # not a parse issue we can rescue — original error stands
            if timezone.is_naive(naive):
                return naive.replace(fold=0, tzinfo=timezone.get_current_timezone())
            raise


class MealForm(forms.ModelForm):
    eaten_at = FoldTolerantDateTimeField(
        initial=timezone.localtime,
        widget=forms.DateTimeInput(
            attrs={"type": "datetime-local"}, format="%Y-%m-%dT%H:%M"
        ),
    )

    class Meta:
        model = Meal
        fields = ["name", "meal_type", "calories", "eaten_at", "notes"]
        widgets = {
            "notes": forms.Textarea(attrs={"rows": 2}),
            "calories": forms.NumberInput(attrs={"min": 1, "max": 5000}),
        }

    def clean_eaten_at(self):
        eaten_at = self.cleaned_data["eaten_at"]
        if eaten_at > timezone.now():
            raise ValidationError("You can't log a meal in the future.")
        return eaten_at
