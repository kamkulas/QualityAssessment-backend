from django.db import models
from django.contrib.postgres.fields import ArrayField

from system.models import System


class Loop(models.Model):
    system = models.ForeignKey(System, on_delete=models.CASCADE, related_name='loops')
    name = models.CharField(max_length=50)
    description = models.TextField(blank=True)
    calculated = models.BooleanField(default=False)
    mv = ArrayField(models.DecimalField(max_digits=15, decimal_places=4))
    ma = ArrayField(models.DecimalField(max_digits=15, decimal_places=4))
    spa = ArrayField(models.DecimalField(max_digits=15, decimal_places=4))
    cv = ArrayField(models.DecimalField(max_digits=15, decimal_places=4))
    ise = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minIse = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    iae = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minIae = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    qe = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minQe = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    skewness = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    kurtosis = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    hre = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minHre = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    hde = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minHde = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    h0 = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    h1 = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    h2 = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    h3 = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    cr1 = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    cr2 = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    gsig = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minGsig = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    salf = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    sgam = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minSgam = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    lb = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minLb = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    rsig = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    minRsig = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    crossX1 = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    crossY1 = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    crossX2 = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    crossY2 = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    gauss = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    levy = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    laplace = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    huber = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    histX = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    histY = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    xx = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    yy = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    haa = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    xp = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    yp = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    firstApproximation = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    secondApproximation = ArrayField(models.DecimalField(max_digits=15, decimal_places=4), blank=True, null=True)
    hausDim = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
    score = models.DecimalField(max_digits=15, decimal_places=4, blank=True, null=True)
