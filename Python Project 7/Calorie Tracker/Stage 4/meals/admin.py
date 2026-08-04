"""Django admin — a free CRUD backoffice for your models.

Run `python manage.py createsuperuser`, then visit /admin/. The options
below control the list page: columns, sidebar filters, search box, and a
date drill-down. This is often the first thing that sells people on Django.
"""

from django.contrib import admin

from .models import Meal


@admin.register(Meal)
class MealAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "meal_type", "calories", "eaten_at")
    list_filter = ("user", "meal_type", "eaten_at")
    search_fields = ("name", "notes")
    date_hierarchy = "eaten_at"
    ordering = ("-eaten_at",)
