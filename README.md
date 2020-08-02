# pylmm
 
Mixed models in python with R-style formulas (via patsy with the addition of lme4 syntax for specifying grouping factors).  

LME2 is the basic linear mixed model class, which defaults to gradient based optimization, with the option to use the hessian.  Both the gradient and hessian are exact (i.e. not numerical approximations), so they can be somewhat time consuming.

LME3 is the same as LME2, except it uses a cholesky parameterization that removes the need for using constraints in optimization.  A totally unconstrained parameterization would be the log cholesky, which is not yet implemented

GLMM implements penalized quasi-likelihood based optimization for generalized mixed models - so far only Binomial and Poisson are functional.  GLMM_AGQ is a work in progress that uses numerical integration, but as of yet only works with one grouping factor and an intercept.  

MixedMCMC is a work in progress, so far only programmed to fit a binary logistic model, but it can do so using either Metropolis/Gibbs sampling or Slice/Gibbs sampling.
