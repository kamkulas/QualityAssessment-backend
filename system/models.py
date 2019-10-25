from django.db import models


class System(models.Model):
    name = models.CharField(max_length=50)
    description = models.CharField(max_length=255)
    filename = models.CharField(max_length=255)
