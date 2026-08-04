"""Serializers — DRF's forms-for-JSON.

A ModelSerializer mirrors ModelForm: fields, labels, and validation derive
from the model (including the calories Min/Max validators — DRF copies model
field validators onto generated serializer fields automatically). You add
what the model can't know, exactly like Stage 1's form did: the
"not in the future" rule, via validate_<fieldname>().

`user` is deliberately NOT a serializer field: ownership is stamped
server-side in the ViewSet's perform_create — never trusted from the client
payload. Same principle as Stage 2's form.save(commit=False).
"""

from django.utils import timezone
from rest_framework import serializers

from meals.models import Meal


class AwareDateTimeField(serializers.DateTimeField):
    """Reject naive datetimes instead of silently assuming server timezone.

    Without this, eaten_at="2026-08-03T12:00:00" (no offset) would be
    interpreted as America/New_York — silently shifting data for any
    client in another timezone. Date-only strings (naive midnight) are
    rejected for the same reason. The API contract: always send an offset.
    """

    def enforce_timezone(self, value):
        if timezone.is_naive(value):
            raise serializers.ValidationError(
                "eaten_at must include a UTC offset, e.g. 2026-08-03T12:00:00-04:00."
            )
        return super().enforce_timezone(value)


class MealSerializer(serializers.ModelSerializer):
    eaten_at = AwareDateTimeField()

    class Meta:
        model = Meal
        fields = ["id", "name", "meal_type", "calories", "eaten_at", "notes", "created_at"]
        read_only_fields = ["id", "created_at"]

    def validate_eaten_at(self, value):
        if value > timezone.now():
            raise serializers.ValidationError("You can't log a meal in the future.")
        return value


class DailySummarySerializer(serializers.Serializer):
    """Read-only shape for the /api/meals/summary/ aggregation rows."""

    day = serializers.DateField()
    total = serializers.IntegerField()
    count = serializers.IntegerField()
