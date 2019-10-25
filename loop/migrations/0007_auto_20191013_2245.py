# Generated by Django 2.2.5 on 2019-10-13 22:45

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('loop', '0006_auto_20191013_2222'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='loop',
            name='crossX1',
        ),
        migrations.RemoveField(
            model_name='loop',
            name='crossY1',
        ),
        migrations.RemoveField(
            model_name='loop',
            name='crossX2',
        ),
        migrations.RemoveField(
            model_name='loop',
            name='crossY2',
        ),
        migrations.AddField(
            model_name='loop',
            name='crossX1',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=4, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AddField(
            model_name='loop',
            name='crossX2',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=4, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AddField(
            model_name='loop',
            name='crossY1',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=4, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AddField(
            model_name='loop',
            name='crossY2',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=4, max_digits=15), blank=True, null=True, size=None),
        ),
    ]
