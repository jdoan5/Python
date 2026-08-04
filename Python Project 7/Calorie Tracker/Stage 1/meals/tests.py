"""Tests — Django's TestCase gives every test a clean, rolled-back DB.

Run:  python manage.py test

Django testing notes:
- TestCase wraps each test in a transaction and rolls it back — tests never
  see each other's data and never touch your real db.sqlite3.
- self.client is a fake browser: .get()/.post() run the full URL → view →
  template cycle without a server.
- reverse('meals:today') resolves URL names, so tests don't hardcode paths.
"""

from datetime import datetime, time, timedelta

from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .forms import MealForm
from .models import Meal


def local_noon(days_ago=0):
    """An aware datetime at noon, `days_ago` local days back.

    Tests anchor to noon instead of offsets from now(): a meal created at
    "now minus 2 hours" lands on YESTERDAY's local date when the suite runs
    just after midnight, which made the totals tests flaky between 00:00 and
    02:00. Noon minus any offset used below stays inside the same local day.
    """
    d = timezone.localdate() - timedelta(days=days_ago)
    return timezone.make_aware(datetime.combine(d, time(12, 0)))


def make_meal(name="Test meal", calories=500, hours_ago=1, **kwargs):
    return Meal.objects.create(
        name=name,
        calories=calories,
        eaten_at=local_noon() - timedelta(hours=hours_ago),
        **kwargs,
    )


class MealModelTests(TestCase):
    def test_str_shows_name_and_calories(self):
        meal = make_meal(name="Pho", calories=550)
        self.assertEqual(str(meal), "Pho (550 kcal)")

    def test_default_ordering_is_newest_first(self):
        older = make_meal(name="older", hours_ago=5)
        newer = make_meal(name="newer", hours_ago=1)
        self.assertEqual(list(Meal.objects.all()), [newer, older])


class MealFormTests(TestCase):
    def valid_data(self, **overrides):
        data = {
            "name": "Lunch bowl",
            "meal_type": "lunch",
            "calories": 600,
            "eaten_at": (timezone.localtime() - timedelta(hours=1)).strftime(
                "%Y-%m-%dT%H:%M"
            ),
            "notes": "",
        }
        data.update(overrides)
        return data

    def test_valid_form(self):
        form = MealForm(data=self.valid_data())
        self.assertTrue(form.is_valid(), form.errors)

    def test_future_meal_rejected(self):
        future = (timezone.localtime() + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M")
        form = MealForm(data=self.valid_data(eaten_at=future))
        self.assertFalse(form.is_valid())
        self.assertIn("eaten_at", form.errors)

    def test_zero_calories_rejected(self):
        form = MealForm(data=self.valid_data(calories=0))
        self.assertFalse(form.is_valid())
        self.assertIn("calories", form.errors)

    def test_absurd_calories_rejected(self):
        form = MealForm(data=self.valid_data(calories=99999))
        self.assertFalse(form.is_valid())
        self.assertIn("calories", form.errors)

    def test_dst_fallback_hour_is_loggable(self):
        # 1:30 AM on the November fall-back night occurs twice; stock Django
        # rejects it as "ambiguous". Our FoldTolerantDateTimeField resolves it
        # to the first occurrence instead — a real 1:30 AM snack must be
        # loggable. (2025-11-02 is a past US DST transition.)
        form = MealForm(data=self.valid_data(eaten_at="2025-11-02T01:30"))
        self.assertTrue(form.is_valid(), form.errors)


@override_settings(DAILY_CALORIE_GOAL=2000)
class TodayViewTests(TestCase):
    def test_page_renders_with_zero_state(self):
        response = self.client.get(reverse("meals:today"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Nothing logged yet")

    def test_running_total_sums_only_todays_meals(self):
        make_meal(calories=300, hours_ago=1)
        make_meal(calories=200, hours_ago=2)
        # Yesterday's meal must not count toward today's total.
        Meal.objects.create(
            name="yesterday", calories=999, eaten_at=local_noon(days_ago=1),
        )
        response = self.client.get(reverse("meals:today"))
        self.assertEqual(response.context["total"], 500)

    def test_post_valid_meal_redirects_and_saves(self):
        response = self.client.post(
            reverse("meals:today"),
            {
                "name": "Banh mi",
                "meal_type": "lunch",
                "calories": 550,
                "eaten_at": (timezone.localtime() - timedelta(minutes=30)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "notes": "",
            },
        )
        # PRG pattern: successful POST redirects (302), never renders directly.
        self.assertRedirects(response, reverse("meals:today"))
        self.assertEqual(Meal.objects.count(), 1)

    def test_post_invalid_meal_rerenders_with_errors(self):
        response = self.client.post(
            reverse("meals:today"),
            {"name": "", "meal_type": "lunch", "calories": 0, "eaten_at": "", "notes": ""},
        )
        self.assertEqual(response.status_code, 200)  # re-render, not redirect
        self.assertEqual(Meal.objects.count(), 0)
        self.assertTrue(response.context["form"].errors)

    def test_over_goal_flag(self):
        make_meal(calories=2500)
        response = self.client.get(reverse("meals:today"))
        self.assertTrue(response.context["over_goal"])
        self.assertEqual(response.context["over_by"], 500)
        self.assertEqual(response.context["percent"], 100)  # capped for the bar


class HistoryViewTests(TestCase):
    def test_late_night_meal_belongs_to_its_local_day(self):
        # A 11:30 PM meal must group under THAT local date, not roll into the
        # next day (UTC storage would put 11:30 PM EDT at 03:30 UTC tomorrow —
        # TruncDate must use the current timezone, not UTC).
        late = local_noon(days_ago=1) + timedelta(hours=11, minutes=30)  # 23:30 yesterday
        Meal.objects.create(name="midnight snack", calories=250, eaten_at=late)
        response = self.client.get(reverse("meals:history"))
        days = list(response.context["days"])
        self.assertEqual(len(days), 1)
        self.assertEqual(days[0]["day"], timezone.localdate() - timedelta(days=1))

    def test_groups_by_day_with_totals(self):
        make_meal(calories=300, hours_ago=1)
        make_meal(calories=200, hours_ago=2)
        Meal.objects.create(
            name="yesterday", calories=400, eaten_at=local_noon(days_ago=1),
        )
        response = self.client.get(reverse("meals:history"))
        days = list(response.context["days"])
        self.assertEqual(len(days), 2)
        self.assertEqual(days[0]["total"], 500)   # newest day first
        self.assertEqual(days[0]["count"], 2)
        self.assertEqual(days[1]["total"], 400)


class DayDetailViewTests(TestCase):
    def test_valid_date_shows_meals(self):
        meal = make_meal(calories=350)
        day = timezone.localdate().isoformat()
        response = self.client.get(reverse("meals:day_detail", args=[day]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, meal.name)
        self.assertEqual(response.context["total"], 350)

    def test_garbage_date_is_404_not_500(self):
        # The isodate path converter refuses these before any view runs.
        for bad in ("not-a-date", "2026-13-45", "20260804", "2026-W32-2"):
            response = self.client.get(f"/history/{bad}/")
            self.assertEqual(response.status_code, 404, bad)


class SeedCommandTests(TestCase):
    def test_seed_creates_meals_and_is_idempotent_per_day(self):
        call_command("seed_meals", days=3, verbosity=0)
        first_count = Meal.objects.count()
        self.assertGreater(first_count, 0)
        call_command("seed_meals", days=3, verbosity=0)
        self.assertEqual(Meal.objects.count(), first_count)  # skips seeded days
