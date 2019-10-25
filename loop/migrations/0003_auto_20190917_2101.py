# Generated by Django 2.2.5 on 2019-09-17 21:01

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('loop', '0002_auto_20190916_2142'),
    ]

    operations = [
        migrations.AlterField(
            model_name='loop',
            name='cr1',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='cr2',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='cv',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='gauss',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='h0',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='h1',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='h2',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='h3',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='h4',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='hde',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='histX',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='histY',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='hre',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='huber',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='iae',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='ise',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='laplace',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='levy',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), blank=True, null=True, size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='ma',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='mu',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='mv',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='qe',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
        migrations.AlterField(
            model_name='loop',
            name='spa',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.DecimalField(decimal_places=11, max_digits=15), size=None),
        ),
        migrations.AlterField(
            model_name='loop',
            name='st',
            field=models.DecimalField(blank=True, decimal_places=11, max_digits=15, null=True),
        ),
    ]
