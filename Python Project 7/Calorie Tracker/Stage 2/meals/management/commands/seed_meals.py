"""`python manage.py seed_meals` — fill the DB with a week of demo data.

Custom management commands are Django's answer to "I need a script that can
use my models": they run inside the app context with settings + ORM loaded.
"""

import random
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from meals.models import Meal

SAMPLES = {
    Meal.MealType.BREAKFAST: [("Oatmeal with banana", 320), ("Eggs and toast", 410), ("Greek yogurt + granola", 280)],
    Meal.MealType.LUNCH: [("Chicken rice bowl", 620), ("Turkey sandwich", 480), ("Pho", 550)],
    Meal.MealType.DINNER: [("Salmon + vegetables", 580), ("Spaghetti bolognese", 720), ("Stir-fry tofu", 490)],
    Meal.MealType.SNACK: [("Apple", 95), ("Protein bar", 210), ("Trail mix", 180)],
}


class Command(BaseCommand):
    help = "Seed one user's account with demo meals (idempotent per day+meal slot)."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=7, help="How many past days to seed.")
        parser.add_argument("--username", required=True,
                            help="Which user receives the seeded meals.")

    def handle(self, *args, **options):
        User = get_user_model()
        try:
            user = User.objects.get(username=options["username"])
        except User.DoesNotExist:
            raise CommandError(f"No user named {options['username']!r} — create one first.")
        created = 0
        now = timezone.localtime()
        for offset in range(options["days"]):
            day = now - timedelta(days=offset)
            for meal_type, hour in [
                (Meal.MealType.BREAKFAST, 8),
                (Meal.MealType.LUNCH, 12),
                (Meal.MealType.DINNER, 19),
                (Meal.MealType.SNACK, 15),
            ]:
                eaten = day.replace(hour=hour, minute=random.randint(0, 59))
                if eaten > timezone.localtime():
                    continue  # don't seed the future on today's row
                # Idempotency is per (day, meal_type) slot — a day-level check
                # would permanently skip days that were only partially seeded
                # (e.g. a morning run creates breakfast only, then "has data").
                if Meal.objects.filter(
                    user=user, eaten_at__date=day.date(), meal_type=meal_type
                ).exists():
                    continue
                name, calories = random.choice(SAMPLES[meal_type])
                Meal.objects.create(
                    user=user, name=name, meal_type=meal_type, calories=calories, eaten_at=eaten
                )
                created += 1
        if options["verbosity"] >= 1:  # honor -v 0 (tests pass verbosity=0)
            self.stdout.write(self.style.SUCCESS(f"Created {created} meals."))
