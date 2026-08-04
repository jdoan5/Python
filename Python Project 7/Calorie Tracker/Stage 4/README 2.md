# Calorie Tracker — Stage 2 (Users & Auth)

Stage 1's tracker, now multi-user: sign up, sign in, and see **only your
own** meals and history. This is the auth concept set every Django interview
touches: gates, scoping, and the built-in auth system.

## Run it

```bash
cd "Stage 2"
source .venv/bin/activate
python manage.py migrate
python manage.py createsuperuser            # you'll want /admin/ anyway
python manage.py seed_meals --username <your-username>
python manage.py runserver                  # http://127.0.0.1:8000
```

Visiting `/` anonymously now redirects to `/accounts/login/?next=/` — sign
up at `/accounts/signup/`, or sign in. Tests:

```bash
python manage.py test                       # 21 tests
```

## What Stage 2 adds (and the concepts behind it)

| Change | Concept |
|---|---|
| `Meal.user = ForeignKey(settings.AUTH_USER_MODEL, on_delete=CASCADE, related_name="meals")` | model relations; always reference the user model indirectly; cascade semantics; `user.meals.all()` |
| `@login_required` / `LoginRequiredMixin` | the auth **gate**: anonymous → `LOGIN_URL` with `?next=` round-trip |
| `Meal.objects.filter(user=request.user)` in every view | the auth **scope** — forgetting this on one query is how data leaks ship; isolation tests assert it per view |
| `form.save(commit=False)` → `meal.user = request.user` → `save()` | stamping ownership server-side (never trust a form field for it) |
| `include("django.contrib.auth.urls")` | free login/logout/password views; you supply `templates/registration/login.html` |
| `accounts` app with `SignUpView(CreateView)` + `UserCreationForm` | the one auth view Django doesn't ship; auto-login on signup via `form_valid()` |
| POST logout form in the nav | Django ≥5 removed GET logout (CSRF-safe state change) |
| `LOGIN_URL` / `LOGIN_REDIRECT_URL` / `LOGOUT_REDIRECT_URL` | the three settings that wire the flow together |
| `seed_meals --username` | management commands with required args + `CommandError` |
| `client.force_login()` in tests | fast authenticated test client; `client.login()` only when testing the form |

## The two-line security model

Every view answers two questions, in order:
1. **Who are you?** (gate — redirect if unknown)
2. **What's yours?** (scope — filter every queryset by `request.user`)

The isolation tests (`test_total_excludes_other_users_meals`,
`test_history_excludes_other_users`, `test_valid_date_shows_own_meals_only`)
exist because scoping bugs are invisible in single-user manual testing —
you need a second user to notice, and tests are the cheapest second user.

## Next: Stage 3

Django REST Framework API (per-user tokens), daily-trend charts, CSV export.
