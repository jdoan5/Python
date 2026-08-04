# Calorie Tracker — Stage 1 (Core Django)

Log meals, see a live running total against a daily calorie goal, and browse
day-by-day history. Built to learn Django's core loop: **URL → view → ORM →
template**, plus forms, migrations, the admin, and tests.

## Run it

```bash
cd "Stage 1"
source .venv/bin/activate               # venv already created (Python 3.14)
python manage.py migrate                # create db.sqlite3 + tables
python manage.py seed_meals             # optional: a week of demo data
python manage.py runserver              # http://127.0.0.1:8000
```

Admin backoffice (optional but worth seeing):

```bash
python manage.py createsuperuser        # then visit /admin/
```

Tests:

```bash
python manage.py test                   # 15 tests, isolated throwaway DB
```

## PyCharm setup

1. Open the `Stage 1` folder as the project.
2. Settings → Project → Python Interpreter → Add Existing → `.venv/bin/python`.
3. Run config: **+ → Python** → module name `manage`, parameters `runserver`,
   working directory = `Stage 1`. (PyCharm Professional has a dedicated
   Django run config; Community works fine with this.)
4. A second config with parameters `test` gives you one-click test runs.

## What each piece teaches (the Django mental model)

| File | Concept |
|---|---|
| `config/settings.py` | one settings module; apps registered in `INSTALLED_APPS`; env-based `SECRET_KEY`/`DEBUG`; custom setting `DAILY_CALORIE_GOAL` |
| `config/urls.py` → `meals/urls.py` | URL routing with `include()` + namespaced names (`meals:today`) |
| `meals/models.py` | the ORM: fields, choices (`TextChoices`), validators, `Meta.ordering`; `USE_TZ` timezone handling |
| `meals/migrations/` | schema as versioned code — `makemigrations` writes them, `migrate` applies them |
| `meals/forms.py` | `ModelForm` + `clean_<field>()` custom validation ("no future meals") |
| `meals/views.py` | FBVs vs CBVs side by side; the **Post/Redirect/Get** pattern; DB-side aggregation with `Sum`/`Count`/`TruncDate` |
| `templates/` | template inheritance (`base.html` + blocks), `{% url %}`, `{% csrf_token %}`, filters |
| `meals/admin.py` | the free CRUD backoffice: `list_display`, filters, search, date drill-down |
| `meals/management/commands/seed_meals.py` | custom `manage.py` commands |
| `meals/tests.py` | `TestCase` transactional isolation, the test `Client`, `reverse()`, `override_settings` |

## Design notes

- **Running totals are computed in the database**, not Python:
  `Meal.objects.filter(eaten_at__date=today).aggregate(Sum("calories"))`.
  The history page is one `GROUP BY` query via `annotate(day=TruncDate(...))`.
- **Timezone correctness**: `USE_TZ=True` stores UTC; `TIME_ZONE =
  "America/New_York"` makes `eaten_at__date` and `timezone.localdate()`
  agree with the user's wall clock. A meal logged at 11:30 PM belongs to
  *that* day — the tests pin this.
- **Validation lives in the form layer** (Django's convention): model
  validators define the bounds; the form enforces them on input and adds
  the can't-log-the-future rule.
- `DAILY_CALORIE_GOAL` is a setting (overridable via env var) rather than a
  hardcoded number — and `override_settings` in tests shows why that's the
  testable choice.
