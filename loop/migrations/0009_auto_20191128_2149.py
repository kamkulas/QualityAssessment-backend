# Generated by Django 2.2.6 on 2019-11-28 21:49

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('loop', '0008_auto_20191105_2245'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='loop',
            name='minCr1',
        ),
        migrations.RemoveField(
            model_name='loop',
            name='minCr2',
        ),
    ]