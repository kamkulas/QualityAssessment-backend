from django.contrib import admin
from loop.models import Loop


@admin.register(Loop)
class LoopAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'system_name')
    
    def system_name(self, obj):
        return obj.system.name
