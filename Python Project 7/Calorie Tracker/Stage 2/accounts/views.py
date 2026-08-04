"""Signup — the one auth view Django doesn't ship.

django.contrib.auth provides login, logout, and password views out of the
box (wired in config/urls.py via `include("django.contrib.auth.urls")`);
registration is intentionally left to you. CreateView + UserCreationForm is
the canonical minimal version; form_valid() logs the new user straight in so
signup lands on the app, not the login page.
"""

from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView


class SignUpView(CreateView):
    form_class = UserCreationForm
    template_name = "registration/signup.html"
    success_url = reverse_lazy("meals:today")

    def dispatch(self, request, *args, **kwargs):
        # Already signed in? Don't offer signup — POSTing here would create a
        # second account and silently switch the session to it. (Mirrors
        # LoginView's redirect_authenticated_user behavior.)
        if request.user.is_authenticated:
            return redirect("meals:today")
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        response = super().form_valid(form)  # saves the user (self.object)
        login(self.request, self.object)
        return response
