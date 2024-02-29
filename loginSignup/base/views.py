from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout

@login_required
def home(request):
    return render(request, "home.html", {})

def authView(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST or None)
        if form.is_valid():
            form.save()
            return redirect("base:login")
    else:
        form = UserCreationForm()
    return render(request, "registration/signup.html", {"form": form})

def custom_logout(request):
    logout(request)
    # Redirect to the login page after logout.
    return redirect('base:login')
