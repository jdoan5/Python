"""Stage 2 tests — everything Stage 1 covered, now under authentication,
plus the tests that matter most in multi-user apps: ISOLATION.

New Django concepts exercised here:
- self.client.force_login(user): skip the login form, authenticate the test
  client directly (fast; use client.login() when testing the form itself).
- assertRedirects with the ?next= login round-trip.
- Per-user data isolation asserted on EVERY view — the tests that catch
  the missing-filter bug before a user does.
"""

from datetime import datetime, time, timedelta

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .forms import MealForm
from .models import Meal

User = get_user_model()


def local_noon(days_ago=0):
    """Aware datetime at local noon `days_ago` back — anchored so tests are
    deterministic even when the suite runs just after midnight."""
    d = timezone.localdate() - timedelta(days=days_ago)
    return timezone.make_aware(datetime.combine(d, time(12, 0)))


def make_user(username="alice"):
    return User.objects.create_user(username=username, password="test-pass-123")


def make_meal(user, name="Test meal", calories=500, hours_ago=1, **kwargs):
    return Meal.objects.create(
        user=user,
        name=name,
        calories=calories,
        eaten_at=local_noon() - timedelta(hours=hours_ago),
        **kwargs,
    )


class MealModelTests(TestCase):
    def setUp(self):
        self.user = make_user()

    def test_str_shows_name_and_calories(self):
        meal = make_meal(self.user, name="Pho", calories=550)
        self.assertEqual(str(meal), "Pho (550 kcal)")

    def test_default_ordering_is_newest_first(self):
        older = make_meal(self.user, name="older", hours_ago=5)
        newer = make_meal(self.user, name="newer", hours_ago=1)
        self.assertEqual(list(Meal.objects.all()), [newer, older])

    def test_related_name_gives_user_meals(self):
        make_meal(self.user)
        self.assertEqual(self.user.meals.count(), 1)

    def test_deleting_user_cascades_to_meals(self):
        make_meal(self.user)
        self.user.delete()
        self.assertEqual(Meal.objects.count(), 0)


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
        self.assertTrue(MealForm(data=self.valid_data()).is_valid())

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
        form = MealForm(data=self.valid_data(eaten_at="2025-11-02T01:30"))
        self.assertTrue(form.is_valid(), form.errors)


class AuthGateTests(TestCase):
    """Anonymous users are redirected to login with ?next= round-trip."""

    def test_all_views_require_login(self):
        day = timezone.localdate().isoformat()
        for url in (
            reverse("meals:today"),
            reverse("meals:history"),
            reverse("meals:day_detail", args=[day]),
        ):
            response = self.client.get(url)
            self.assertRedirects(response, f"{reverse('login')}?next={url}")


class SignupTests(TestCase):
    def test_signup_creates_user_and_logs_in(self):
        response = self.client.post(
            reverse("accounts:signup"),
            {
                "username": "newuser",
                "password1": "a-strong-pass-9",
                "password2": "a-strong-pass-9",
            },
        )
        self.assertRedirects(response, reverse("meals:today"))
        self.assertTrue(User.objects.filter(username="newuser").exists())
        # The signup view logged the new user in — today page loads, no redirect.
        self.assertEqual(self.client.get(reverse("meals:today")).status_code, 200)

    def test_authenticated_user_redirected_away_from_signup(self):
        user = make_user("existing")
        self.client.force_login(user)
        self.assertRedirects(self.client.get(reverse("accounts:signup")),
                             reverse("meals:today"))
        # POST must not create a second account or switch the session.
        response = self.client.post(
            reverse("accounts:signup"),
            {"username": "sneaky", "password1": "a-strong-pass-9",
             "password2": "a-strong-pass-9"},
        )
        self.assertRedirects(response, reverse("meals:today"))
        self.assertFalse(User.objects.filter(username="sneaky").exists())
        self.assertEqual(int(self.client.session["_auth_user_id"]), user.pk)

    def test_mismatched_passwords_rerender(self):
        response = self.client.post(
            reverse("accounts:signup"),
            {"username": "x", "password1": "abc12345!", "password2": "different"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="x").exists())


@override_settings(DAILY_CALORIE_GOAL=2000)
class TodayViewTests(TestCase):
    def setUp(self):
        self.user = make_user()
        self.client.force_login(self.user)

    def test_page_renders_with_zero_state(self):
        response = self.client.get(reverse("meals:today"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Nothing logged yet")
        self.assertEqual(response.context["total"], 0)

    def test_running_total_sums_only_todays_meals(self):
        make_meal(self.user, calories=300, hours_ago=1)
        make_meal(self.user, calories=200, hours_ago=2)
        Meal.objects.create(
            user=self.user, name="yesterday", calories=999,
            eaten_at=local_noon(days_ago=1),
        )
        response = self.client.get(reverse("meals:today"))
        self.assertEqual(response.context["total"], 500)

    def test_total_excludes_other_users_meals(self):
        other = make_user("bob")
        make_meal(other, calories=1800)
        make_meal(self.user, calories=300)
        response = self.client.get(reverse("meals:today"))
        self.assertEqual(response.context["total"], 300)  # bob's meal invisible

    def test_post_stamps_the_logged_in_user(self):
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
        self.assertRedirects(response, reverse("meals:today"))
        meal = Meal.objects.get()
        self.assertEqual(meal.user, self.user)

    def test_post_invalid_meal_rerenders_with_errors(self):
        response = self.client.post(
            reverse("meals:today"),
            {"name": "", "meal_type": "lunch", "calories": 0, "eaten_at": "", "notes": ""},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Meal.objects.count(), 0)
        self.assertTrue(response.context["form"].errors)

    def test_over_goal_flag(self):
        make_meal(self.user, calories=2500)
        response = self.client.get(reverse("meals:today"))
        self.assertTrue(response.context["over_goal"])
        self.assertEqual(response.context["over_by"], 500)
        self.assertEqual(response.context["percent"], 100)  # capped for the bar


class HistoryViewTests(TestCase):
    def setUp(self):
        self.user = make_user()
        self.client.force_login(self.user)

    def test_groups_by_day_with_totals(self):
        make_meal(self.user, calories=300, hours_ago=1)
        make_meal(self.user, calories=200, hours_ago=2)
        Meal.objects.create(
            user=self.user, name="yesterday", calories=400,
            eaten_at=local_noon(days_ago=1),
        )
        response = self.client.get(reverse("meals:history"))
        days = list(response.context["days"])
        self.assertEqual(len(days), 2)
        self.assertEqual(days[0]["total"], 500)
        self.assertEqual(days[1]["total"], 400)

    def test_history_excludes_other_users(self):
        other = make_user("bob")
        make_meal(other, calories=800)
        response = self.client.get(reverse("meals:history"))
        self.assertEqual(list(response.context["days"]), [])

    def test_late_night_meal_belongs_to_its_local_day(self):
        late = local_noon(days_ago=1) + timedelta(hours=11, minutes=30)
        Meal.objects.create(user=self.user, name="midnight snack",
                            calories=250, eaten_at=late)
        response = self.client.get(reverse("meals:history"))
        days = list(response.context["days"])
        self.assertEqual(days[0]["day"], timezone.localdate() - timedelta(days=1))


class DayDetailViewTests(TestCase):
    def setUp(self):
        self.user = make_user()
        self.client.force_login(self.user)

    def test_valid_date_shows_own_meals_only(self):
        mine = make_meal(self.user, name="my lunch", calories=350)
        other = make_user("bob")
        make_meal(other, name="bobs lunch", calories=700)
        day = timezone.localdate().isoformat()
        response = self.client.get(reverse("meals:day_detail", args=[day]))
        self.assertContains(response, mine.name)
        self.assertNotContains(response, "bobs lunch")
        self.assertEqual(response.context["total"], 350)

    def test_garbage_date_is_404_not_500(self):
        for bad in ("not-a-date", "2026-13-45", "20260804", "2026-W32-2"):
            self.assertEqual(self.client.get(f"/history/{bad}/").status_code, 404, bad)


class SeedCommandTests(TestCase):
    def test_seed_requires_existing_user(self):
        with self.assertRaises(CommandError):
            call_command("seed_meals", username="ghost", verbosity=0)

    def test_seed_scopes_to_named_user(self):
        alice, bob = make_user("alice2"), make_user("bob2")
        call_command("seed_meals", days=3, username="alice2", verbosity=0)
        self.assertGreater(alice.meals.count(), 0)
        self.assertEqual(bob.meals.count(), 0)
        # Idempotent per user
        count = alice.meals.count()
        call_command("seed_meals", days=3, username="alice2", verbosity=0)
        self.assertEqual(alice.meals.count(), count)
