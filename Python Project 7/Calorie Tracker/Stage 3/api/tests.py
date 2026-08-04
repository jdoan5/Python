"""API tests — DRF's APITestCase/APIClient speak JSON natively.

The API repeats Stage 2's two questions (gate, scope) in a new dialect:
- gate: unauthenticated requests get 401/403, token auth via the
  Authorization header, session auth for the browser.
- scope: cross-user object access must be 404 — the object isn't in your
  queryset, so the API behaves as if it doesn't exist (leaks less than 403,
  which would confirm the ID exists).
"""

from datetime import datetime, time, timedelta

from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from meals.models import Meal

User = get_user_model()


def local_noon(days_ago=0):
    d = timezone.localdate() - timedelta(days=days_ago)
    return timezone.make_aware(datetime.combine(d, time(12, 0)))


def make_user(username="alice"):
    return User.objects.create_user(username=username, password="test-pass-123")


def make_meal(user, name="Test meal", calories=500, days_ago=0, **kwargs):
    return Meal.objects.create(
        user=user, name=name, calories=calories,
        eaten_at=local_noon(days_ago=days_ago) - timedelta(hours=1), **kwargs,
    )


def payload(**overrides):
    data = {
        "name": "Banh mi",
        "meal_type": "lunch",
        "calories": 550,
        "eaten_at": (timezone.localtime() - timedelta(minutes=30)).isoformat(),
        "notes": "",
    }
    data.update(overrides)
    return data


class AuthTests(APITestCase):
    def setUp(self):
        # Throttle counters live in the cache and would otherwise bleed
        # between tests (same fake IP for every test-client request).
        from django.core.cache import cache
        cache.clear()

    def test_anonymous_requests_rejected(self):
        for url in (reverse("api:meal-list"), reverse("api:meal-summary")):
            response = self.client.get(url)
            self.assertIn(response.status_code,
                          (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_token_flow_end_to_end(self):
        make_user("carol")
        # 1. Exchange credentials for a token
        response = self.client.post(
            reverse("api:token"),
            {"username": "carol", "password": "test-pass-123"},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        token = response.data["token"]
        # 2. Use it via the Authorization header — no session involved
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token}")
        response = self.client.post(reverse("api:meal-list"), payload())
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_wrong_password_gets_no_token(self):
        make_user("carol")
        response = self.client.post(
            reverse("api:token"), {"username": "carol", "password": "wrong"}
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Token.objects.count(), 0)

    def test_token_endpoint_throttled_after_five_attempts(self):
        make_user("carol")
        url = reverse("api:token")
        for _ in range(5):
            response = self.client.post(url, {"username": "carol", "password": "wrong"})
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        # Sixth guess inside the window: throttled, not evaluated.
        response = self.client.post(url, {"username": "carol", "password": "wrong"})
        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)

    def test_password_change_revokes_token(self):
        user = make_user("dave")
        token = Token.objects.create(user=user)
        self.client.force_login(user)
        response = self.client.post(
            reverse("password_change"),
            {
                "old_password": "test-pass-123",
                "new_password1": "an-even-stronger-8",
                "new_password2": "an-even-stronger-8",
            },
        )
        self.assertEqual(response.status_code, status.HTTP_302_FOUND)
        # The pre-change token must be gone — a stolen-password attacker's
        # token dies with the password.
        self.assertFalse(Token.objects.filter(key=token.key).exists())


class MealCrudTests(APITestCase):
    def setUp(self):
        self.user = make_user()
        self.other = make_user("bob")
        self.client.force_authenticate(self.user)

    def test_create_stamps_authenticated_user(self):
        response = self.client.post(reverse("api:meal-list"), payload())
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Meal.objects.get().user, self.user)

    def test_client_cannot_forge_ownership(self):
        # Even if a malicious payload includes user/user_id, it's ignored —
        # `user` is not a serializer field; perform_create stamps request.user.
        response = self.client.post(
            reverse("api:meal-list"), {**payload(), "user": self.other.pk, "user_id": self.other.pk}
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Meal.objects.get().user, self.user)

    def test_list_returns_own_meals_only(self):
        make_meal(self.user, name="mine")
        make_meal(self.other, name="bobs")
        response = self.client.get(reverse("api:meal-list"))
        names = [m["name"] for m in response.data["results"]]
        self.assertEqual(names, ["mine"])

    def test_list_is_paginated(self):
        make_meal(self.user)
        response = self.client.get(reverse("api:meal-list"))
        for key in ("count", "next", "previous", "results"):
            self.assertIn(key, response.data)

    def test_other_users_meal_is_404_not_403(self):
        theirs = make_meal(self.other)
        for method, args in (("get", ()), ("delete", ()), ("patch", ({"calories": 1},))):
            response = getattr(self.client, method)(
                reverse("api:meal-detail", args=[theirs.pk]), *args
            )
            self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND, method)

    def test_update_and_delete_own_meal(self):
        mine = make_meal(self.user, calories=400)
        response = self.client.patch(
            reverse("api:meal-detail", args=[mine.pk]), {"calories": 450}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mine.refresh_from_db()
        self.assertEqual(mine.calories, 450)
        response = self.client.delete(reverse("api:meal-detail", args=[mine.pk]))
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(Meal.objects.count(), 0)


class ValidationTests(APITestCase):
    def setUp(self):
        self.client.force_authenticate(make_user())

    def test_future_meal_rejected(self):
        future = (timezone.localtime() + timedelta(hours=2)).isoformat()
        response = self.client.post(reverse("api:meal-list"), payload(eaten_at=future))
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("eaten_at", response.data)

    def test_naive_datetime_rejected(self):
        # No UTC offset -> ambiguous contract -> explicit 400, never a silent
        # server-timezone guess.
        for naive in ("2026-01-15T12:00:00", "2026-01-15"):
            response = self.client.post(reverse("api:meal-list"), payload(eaten_at=naive))
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST, naive)
            self.assertIn("eaten_at", response.data)

    def test_calorie_bounds_enforced_by_model_validators(self):
        # DRF copies MinValue/MaxValue validators from the model field.
        for bad in (0, 99999):
            response = self.client.post(reverse("api:meal-list"), payload(calories=bad))
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST, bad)
            self.assertIn("calories", response.data)


class SummaryTests(APITestCase):
    def setUp(self):
        self.user = make_user()
        self.client.force_authenticate(self.user)

    def test_summary_aggregates_by_day_ascending(self):
        make_meal(self.user, calories=300)
        make_meal(self.user, calories=200)
        make_meal(self.user, calories=400, days_ago=1)
        response = self.client.get(reverse("api:meal-summary"))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)
        self.assertEqual(response.data[0]["total"], 400)   # oldest first
        self.assertEqual(response.data[1]["total"], 500)
        self.assertEqual(response.data[1]["count"], 2)

    def test_summary_scoped_to_user(self):
        make_meal(make_user("bob"), calories=800)
        response = self.client.get(reverse("api:meal-summary"))
        self.assertEqual(response.data, [])


class CsvExportTests(APITestCase):
    def test_login_required(self):
        response = self.client.get(reverse("meals:export_csv"))
        self.assertEqual(response.status_code, status.HTTP_302_FOUND)

    def test_csv_contains_own_meals_only(self):
        user = make_user()
        make_meal(user, name="my pho", calories=550)
        make_meal(make_user("bob"), name="bobs burger")
        self.client.force_login(user)
        response = self.client.get(reverse("meals:export_csv"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("attachment", response["Content-Disposition"])
        body = response.content.decode("utf-8-sig")  # strips the Excel BOM
        self.assertIn("my pho", body)
        self.assertNotIn("bobs burger", body)
        self.assertTrue(body.startswith("eaten_at,name,meal_type,calories,notes"))
