"""App-level URLconf. `app_name` + include() in config/urls.py gives every
route a namespaced name ("meals:today") usable in redirect() and {% url %}.

The custom path converter pins day URLs to exactly YYYY-MM-DD (Python 3.11+'s
date.fromisoformat also accepts '20260804' and week-dates like '2026-W32-2',
which would otherwise create duplicate URLs for the same page). A converter
whose to_python raises ValueError simply doesn't match — Django returns 404,
so invalid dates like 2026-13-45 never reach the view.
"""

from datetime import date

from django.urls import path, register_converter

from . import views


class ISODateConverter:
    regex = r"\d{4}-\d{2}-\d{2}"

    def to_python(self, value: str) -> date:
        return date.fromisoformat(value)  # ValueError -> no match -> 404

    def to_url(self, value) -> str:
        return value.isoformat() if hasattr(value, "isoformat") else str(value)


register_converter(ISODateConverter, "isodate")

app_name = "meals"

urlpatterns = [
    path("", views.today, name="today"),
    path("history/", views.HistoryView.as_view(), name="history"),
    path("history/<isodate:day>/", views.day_detail, name="day_detail"),
]
