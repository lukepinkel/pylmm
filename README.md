# pylmm
 
Mixed models in python with R-style formulas (via patsy with the addition of lme4 syntax for specifying grouping factors).  

LME2 is the basic linear mixed model class, which defaults to gradient based optimization, with the option to use the hessian.  Both the gradient and hessian are exact (i.e. not numerical approximations), so they can be somewhat time consuming.

GLMM implements penalized quasi-likelihood based optimization for generalized mixed models - so far only Binomial and Poisson are functional.  GLMM_AGQ is a work in progress that uses numerical integration, but as of yet only works with one grouping factor and an intercept.  
