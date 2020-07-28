#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:00:38 2020

@author: lukepinkel
"""
import tqdm
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd # analysis:ignore
from .lmm_chol3 import LME3, make_theta
from ..utilities.linalg_operations import vech, _check_shape
from sksparse.cholmod import cholesky
from ..utilities.trnorm import trnorm
    
def log1p(x):
    return np.log(1+x)



def rtnorm(mu, sd, lower, upper):
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    return sp.stats.truncnorm(a=a, b=b, loc=mu, scale=sd).rvs()
    
    

def logit(x):
    u = np.exp(x)
    return u / (1 + u)
    
def probit(x):
    return sp.stats.norm(0, 1).cdf(x)


def get_u_indices(dims): 
    u_indices = {}
    start=0
    for key, val in dims.items():
        q = val['n_groups']*val['n_vars']
        u_indices[key] = np.arange(start, start+q)
        start+=q
    return u_indices

def wishart_info(dims):
    ws = {}
    for key in dims.keys():
        ws[key] = {}
        q = dims[key]['n_groups']
        k = dims[key]['n_vars']
        nu = q-(k+1)
        ws[key]['q'] = q
        ws[key]['k'] = k
        ws[key]['nu'] = nu
    return ws

def sample_gcov(theta, u, wsinfo, indices, key, priors):
    u_i = u[indices['u'][key]]
    U_i = u_i.reshape(-1, wsinfo[key]['k'], order='C')
    Sg_i =  U_i.T.dot(U_i)
    Gs = sp.stats.invwishart(wsinfo[key]['nu']+priors[key]['n'], 
                             Sg_i+priors[key]['V']).rvs()
    theta[indices['theta'][key]] = vech(Gs)
    return theta

def sample_rcov(theta, y, yhat, wsinfo, priors):
    resid = y - yhat
    sse = resid.T.dot(resid)
    nu = wsinfo['r']['nu'] 
    ss = sp.stats.invgamma((nu+priors['R']['n']), 
                            scale=(sse+priors['R']['V'])).rvs()  
    theta[-1] = ss 
    return theta




class MixedMCMC(LME3):
    
    def __init__(self, formula, data, priors=None):
        super().__init__(formula, data) 
        self.t_init, _ = make_theta(self.dims)
        self.W = sp.sparse.csc_matrix(self.XZ)
        self.wsinfo = wishart_info(self.dims)
        self.y = _check_shape(self.y, 1)
        
        self.indices['u'] = get_u_indices(self.dims)
        self.n_re = self.G.shape[0]
        self.n_fe = self.X.shape[1]
        self.n_lc = self.W.shape[1]
        self.n_ob = self.W.shape[0]
        self.re_mu = np.zeros(self.n_re)
        self.n_params = len(self.t_init)+self.n_fe
        if priors is None:
            priors = dict(R=dict(V=0.500*self.n_ob, n=self.n_ob), id1=dict(V=np.eye(2)*0.001, n=4))
            for level in self.levels:
                Vi = np.eye(self.dims[level]['n_vars'])*0.001
                priors[level] = dict(V=Vi, n=4)
        self.priors = priors
        self.wsinfo['r'] = dict(nu=self.n_ob-2)
        self.offset = np.zeros(self.n_lc)
        self.location = np.zeros(self.n_re+self.n_fe)
    
    def sample_location(self, theta, x1, x2, y):
        s, s2 =  np.sqrt(theta[-1]), theta[-1]
        WtR = self.W.copy().T / s2
        M = WtR.dot(self.W)
        Ginv = self.update_gmat(theta, inverse=True).copy()
        M[-self.n_re:, -self.n_re:] += Ginv
        chol_fac = cholesky(Ginv, ordering_method='natural')
        a_star = chol_fac.solve_Lt(x1, use_LDLt_decomposition=False)
        y_z = y - (self.Z.dot(a_star) + x2 * s)
        ofs = self.offset.copy()
        ofs[-self.n_re:] = a_star
        location = ofs + sp.sparse.linalg.spsolve(M, WtR.dot(y_z))
        return location
    
    def mh_lvar(self, pred, s, z, x_step, u_accept, propC):
        z_prop = x_step * propC + z
        
        mndenom1, mndenom2 = np.exp(z)+1, np.exp(z_prop)+1
        densityl1 = (self.y*z) - np.log(mndenom1) 
        densityl2 = (self.y*z_prop) - np.log(mndenom2)
        densityl1 += sp.stats.norm(pred, s).logpdf(z)
        densityl2 += sp.stats.norm(pred, s).logpdf(z_prop)
        density_diff = densityl2 - densityl1
        accept = (density_diff>u_accept)
        z[accept] = z_prop[accept]
        return z, accept
    
    def sample_theta(self, theta, u, z, pred):
        for key in self.levels:
            theta = sample_gcov(theta.copy(), u, self.wsinfo, self.indices,
                                key, self.priors)
        theta = sample_rcov(theta, z, pred, self.wsinfo, self.priors)
        return theta
    
    def sample_mh_gibbs(self, n_samples, propC=1.0, chain=0, store_z=False):
        param_samples = np.zeros((n_samples, self.n_params))
        acceptances = np.zeros((n_samples, self.n_ob))
        if store_z:
            z_samples = np.zeros((n_samples, self.n_ob))
        
        x_step = sp.stats.norm(0.0, 1.0).rvs((n_samples, self.n_ob))
        x_astr = sp.stats.norm(0.0, 1.0).rvs((n_samples, self.n_ob))
        x_ranf = sp.stats.norm(0.0, 1.0).rvs((n_samples, self.n_re))
        u_accp = np.log(sp.stats.uniform(0, 1).rvs(n_samples, self.n_ob))
        
        location = self.location.copy()
        pred =  self.W.dot(location)
        z = sp.stats.norm(self.y, self.y.var()).rvs()
        theta = self.t_init.copy()
       
        progress_bar = tqdm.tqdm(range(n_samples))
        counter = 1
        for i in progress_bar:
            s2 = theta[-1]
            s = np.sqrt(s2)
            z, accept = self.mh_lvar(pred, s, z, x_step[i], u_accp[i], propC)
            location = self.sample_location(theta, x_ranf[i], x_astr[i], z)
            pred = self.W.dot(location)
            u = location[-self.n_re:]
            theta  = self.sample_theta(theta, u, z, pred)
            
            param_samples[i, self.n_fe:] = theta.copy()
            param_samples[i, :self.n_fe] = location[:self.n_fe]
            acceptances[i] = accept
            if store_z:
                z_samples[i] = z
            counter+=1
            if counter==1000:
                acceptance = np.sum(acceptances)/(float((i+1)*self.n_ob))
                progress_bar.set_description(f"Chain {chain} Acceptance Prob: {acceptance:.4f}")
                counter = 1
        progress_bar.close()
        
        if store_z:
            return param_samples, acceptances, z_samples
        else:
            return param_samples, acceptances
        
    def sample_slice_gibbs(self, n_samples):
        normdist = sp.stats.norm(0.0, 1.0).rvs

        n_pr, n_ob, n_re = self.n_params, self.n_ob, self.n_re
        n_smp = n_samples
        samples = np.zeros((n_smp, n_pr))
        
        x_astr, x_ranf = normdist((n_smp, n_ob)), normdist((n_smp, n_re))
        rexpon = sp.stats.expon(scale=1).rvs((n_smp, n_ob))
        
        location, pred = self.location.copy(), self.W.dot(self.location)
        theta, z = self.t_init.copy(), sp.stats.norm(0, 1).rvs(n_ob)
        
        ix0, ix1 = self.y==0, self.y==1
        v = np.zeros_like(z).astype(float)
        progress_bar = tqdm.tqdm(range(n_smp))
        for i in progress_bar:
            #z, accept = model.mh_lvar(pred, np.sqrt(theta[-1]), z, x_step[i], u_accp[i], propC)
            v[ix1] = z[ix1] - log1p(np.exp(z[ix1]))
            v[ix1]-= rexpon[i][ix1]
            v[ix1] = v[ix1] - log1p(-np.exp(v[ix1]))
            
            v[ix0] = -log1p(np.exp(z[ix0]))
            v[ix0]-= rexpon[i][ix0]
            v[ix0] = log1p(-np.exp(v[ix0])) - v[ix0]
            s = np.sqrt(theta[-1])
            z[ix1] = rtnorm(mu=pred[ix1], sd=s, lower=v[ix1], upper=20)
            z[ix0] = rtnorm(mu=pred[ix0], sd=s, lower=-20, upper=v[ix0])
             
            location = self.sample_location(theta, x_ranf[i], x_astr[i], z)
            pred = self.W.dot(location)
            u = location[-self.n_re:]
            theta  = self.sample_theta(theta, u, z, pred)
            
            samples[i, self.n_fe:] = theta.copy()
            samples[i, :self.n_fe] = location[:self.n_fe]
        
        progress_bar.close()
        return samples
    
    def sample_slice_gibbs2(self, n_samples):
        normdist = sp.stats.norm(0.0, 1.0).rvs

        n_pr, n_ob, n_re = self.n_params, self.n_ob, self.n_re
        n_smp = n_samples
        samples = np.zeros((n_smp, n_pr))
        
        x_astr, x_ranf = normdist((n_smp, n_ob)), normdist((n_smp, n_re))
        rexpon = sp.stats.expon(scale=1).rvs((n_smp, n_ob))
        
        location, pred = self.location.copy(), self.W.dot(self.location)
        theta, z = self.t_init.copy(), sp.stats.norm(0, 1).rvs(n_ob)

        ix0, ix1 = self.y==0, self.y==1
        jv0, jv1 = np.ones(len(ix0)), np.ones(len(ix1))
        v = np.zeros_like(z).astype(float)
        progress_bar = tqdm.tqdm(range(n_smp))
        for i in progress_bar:
            #z, accept = model.mh_lvar(pred, np.sqrt(theta[-1]), z, x_step[i], u_accp[i], propC)
            v[ix1] = z[ix1] - log1p(np.exp(z[ix1]))
            v[ix1]-= rexpon[i][ix1]
            v[ix1] = v[ix1] - log1p(-np.exp(v[ix1]))
            
            v[ix0] = -log1p(np.exp(z[ix0]))
            v[ix0]-= rexpon[i][ix0]
            v[ix0] = log1p(-np.exp(v[ix0])) - v[ix0]
            s = np.sqrt(theta[-1])
            z[ix1] = trnorm(mu=pred[ix1], sd=s*jv1, lb=v[ix1], ub=20*jv1)
            z[ix0] = trnorm(mu=pred[ix0], sd=s*jv0, lb=-20*jv0, ub=v[ix0])
             
            location = self.sample_location(theta, x_ranf[i], x_astr[i], z)
            pred = self.W.dot(location)
            u = location[-self.n_re:]
            theta  = self.sample_theta(theta, u, z, pred)
            
            samples[i, self.n_fe:] = theta.copy()
            samples[i, :self.n_fe] = location[:self.n_fe]
        
        progress_bar.close()
        return samples
                
                
                
                
        
        
        
        
                
                
                
                
                        
                        
                        
                        
                
                


    




    
    
    
    
    
    


