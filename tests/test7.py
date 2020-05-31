#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:34:31 2020

@author: lukepinkel
"""

import timeit # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import seaborn as sns # analysis:ignore
import matplotlib as mpl# analysis:ignore
from ..pylmm.lmm import LME, LME2 # analysis:ignore
from ..pylmm.model_matrices import vech2vec, get_jacmats2, jac2deriv # analysis:ignore
import scipy.sparse as sps # analysis:ignore
import matplotlib.pyplot as plt# analysis:ignore
from sksparse.cholmod import cholesky # analysis:ignore
from .test_data import generate_data # analysis:ignore
from ..utilities.random_corr import vine_corr # analysis:ignore
from ..utilities.linalg_operations import invech, vech, scholesky # analysis:ignore
from ..utilities.special_mats import (kronvec_mat, dmat)# analysis:ignore



formula = "y~x1+x5-1+(1+x2|id1)+(1|id2)+(1+x3+x4|id3)"
model_dict = {}
model_dict['gcov'] = {'id1':invech(np.array([1., 0.2, 1.])),
                      'id2':np.array([[1.0]]),
                      'id3':invech(np.array([1., -0.2, -0.2 , 1., 0.3, 1.]))}

model_dict['ginfo'] = {'id1':dict(n_grp=200, n_per=20),
                       'id2':dict(n_grp=50 , n_per=80),
                       'id3':dict(n_grp=100, n_per=40)}
 
model_dict['mu'] = np.zeros(5)
model_dict['vcov'] = vine_corr(5, 20)
model_dict['beta'] = np.array([2, -2])
model_dict['n_obs'] = 4000
df1, formula1 = generate_data(formula, model_dict, r=0.6**0.5)


model1 = LME(formula1, df1)
model2 = LME2(formula1, df1)
model2._fit()

deriv_mats2 = jac2deriv(model2.jac_mats)
np.allclose(deriv_mats2['id1'].A, model1.deriv_mats['id1'].A)


timeit.timeit("model1.gradient(model1.theta)", globals=globals(), number=1)
timeit.timeit("model2.gradient(model1.theta)", globals=globals(), number=1)

timeit.timeit("model1.hessian(model1.theta)", globals=globals(), number=1)
timeit.timeit("model2.hessian(model1.theta)", globals=globals(), number=1)

jac_mats2 = get_jacmats2(model1.Zs, model1.dims, model1.indices, 
                         model1.g_indices, model1.theta)



G0 = model1.G.copy()
jac_mats = model1.jac_mats
deriv_mats = model1.deriv_mats

model1._fit()

g = model1.gradient(model1.theta)

J_id1_0 = jac_mats['id1'][0].A
J_id1_1 = jac_mats['id1'][1].A
D_id1 = deriv_mats['id1'].A


sns.heatmap(J_id1_0, cmap=plt.cm.Greys)
sns.heatmap(J_id1_1, cmap=plt.cm.Greys)

sns.heatmap(D_id1!=0, cmap=plt.cm.Greys)

start = 0
key = 'id1'
value = model1.dims['id1']
nv, ng =  value['n_vars'], value['n_groups']
Sv_shape = nv, nv
Av_shape = ng, ng
Kv = kronvec_mat(Av_shape, Sv_shape)
Z = model1.Z
Zs = model1.Zs

Ip = sps.csc_matrix(sps.eye(np.product(Sv_shape)))
vecAv = sps.csc_matrix(sps.eye(ng)).reshape((-1, 1), order='F')
D = sps.csc_matrix(Kv.dot(sps.kron(vecAv, Ip)))
Zi = Zs[:, start:start+ng*nv]

sqrd = int(np.sqrt(D.shape[1]))
tmp = sps.csc_matrix(dmat(sqrd))
deriv_mats[key] = D.dot(tmp)
dims = model1.dims
G, Ginv, R, Rinv, V, Vinv, W, s = model1._params_to_model(model1.theta)
XtW = W.T.dot(model1.X)
XtW_inv = np.linalg.inv(XtW)
P = Vinv - np.linalg.multi_dot([W, XtW_inv, W.T])
Py = P.dot(model1.y)
PyPy = np.kron(Py, Py)
vecP = P.reshape((-1,1), order='F')
g = model1.deriv_mats[key].T.dot(vecP)-model1.deriv_mats[key].T.dot(PyPy)
g_indices = model1.g_indices
gix = model1.g_indices['id1']

model1.G.data[gix]
vech2vec(model1.theta[model1.indices['id1']])

Gi = G0[start:start+ng*nv, start:start+ng*nv].copy()

Zi.dot(Gi).dot(Zi.T)
theta_i = model1.theta[model1.indices[key]]
for i in range(len(theta_i)):
    dtheta_i = np.zeros_like(theta_i)
    dtheta_i[i] = 1
    dtheta_i = vech2vec(dtheta_i)
    dGi = G0[start:start+ng*nv, start:start+ng*nv].copy()
    dGi.data[g_indices[key]] = np.tile(dtheta_i, ng)
    dVi = (Zi.dot(dGi).dot(Zi.T)).A









