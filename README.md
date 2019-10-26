Backend part of my application for master's thesis.

Thesis subject: Environment for multicriterial quality assessment of control systems.

For this project to work, there is a need to download certain MATLAB scripts (that are necessary to do the assessment) and paste them into matlab directory - this isn't done by default, because I'm not the autor of these scripts.  
Directory structure has to look like this:
```
+-- matlab  
|   +-- FitFunc  
|   |   +-- fit_ML_laplace.m  
|   +-- hurst  
|   |   +-- MultiFit3.m  
|   +-- matlab  
|   |   +-- LIBRA  
|   |   |   +-- mloclogist.m  
|   |   |   +-- mscalelogist.m  
|   +-- stbl  
|   |   +-- stblfit.m  
|   |   +-- stblpdf.m
```