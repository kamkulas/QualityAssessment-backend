"""qualityassessment URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from graphene_django.views import GraphQLView
from django.views.decorators.csrf import csrf_exempt
from system.views import FileUpload
from loop.views import loop_card, trend_plot, control_error_plot, error_plot, \
    characteristics_plot, rs_plot, histogram_plot

urlpatterns = [
    path('admin/', admin.site.urls),
    path('graphql', csrf_exempt(GraphQLView.as_view(graphiql=True))),
    path('file_upload', csrf_exempt(FileUpload.as_view())),
    path('download_card/<int:id>', loop_card, name='loop_card'),
    path('trend_plot/<int:id>', trend_plot, name='trend_plot'),
    path('control_error_plot/<int:id>', control_error_plot, name='control_error_plot'),
    path('error_plot/<int:id>', error_plot, name='error_plot'),
    path('characteristics_plot/<int:id>', characteristics_plot, name='characteristics_plot'),
    path('rs_plot/<int:id>', rs_plot, name='rs_plot'),
    path('histogram_plot/<int:id>', histogram_plot, name='histogram_plot'),
]
