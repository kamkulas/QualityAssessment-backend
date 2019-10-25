from django.contrib import admin
from system.models import System


@admin.register(System)
class SystemAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'description', 'filename')

