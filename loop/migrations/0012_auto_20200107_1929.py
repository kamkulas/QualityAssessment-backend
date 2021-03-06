# Generated by Django 2.2.6 on 2020-01-07 19:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('loop', '0011_loop_description'),
    ]

    operations = [
        migrations.AddField(
            model_name='loop',
            name='kurtosis',
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=15, null=True),
        ),
        migrations.AddField(
            model_name='loop',
            name='minKurtosis',
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=15, null=True),
        ),
        migrations.AddField(
            model_name='loop',
            name='minSkewness',
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=15, null=True),
        ),
        migrations.AddField(
            model_name='loop',
            name='skewness',
            field=models.DecimalField(blank=True, decimal_places=4, max_digits=15, null=True),
        ),
    ]
