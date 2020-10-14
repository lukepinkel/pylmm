# pylmm
 
Mixed models in python with R-style formulas (via patsy with the addition of lme4 syntax for specifying grouping factors).  

LMEC form lmm_chol.py is the basic linear mixed model class, which defaults to gradient based optimization, with the option to use the hessian.  Both the gradient and hessian are exact (i.e. not numerical approximations), so they can be somewhat time consuming.  LMEC uses a cholesky parameterization that makes optimization more stable; however specifying bounds for diagonal variance components are still generally required; A totally unconstrained parameterization would be the log cholesky, which is not yet implemented

GLMM implements penalized quasi-likelihood based optimization for generalized mixed models - so far only Binomial and Poisson are functional.  GLMM_AGQ is a work in progress that uses numerical integration, but as of yet only works with one grouping factor and an intercept.  

MixedMCMC is a work in progress that can fit Bernoulli (binary) logistic models, and normal mixed models.  All the basic machinery is in place for extension (e.g. untested poisson GLMM sampling) but only logistic and normal models have been tested, the former using either Metropolis/Gibbs sampling or Slice/Gibbs sampling, and the latter using Gibbs sampling.
