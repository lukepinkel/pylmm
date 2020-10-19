#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:40:17 2020

@author: lukepinkel
"""

import numpy as np
import scipy as sp
def fo_fc_fd(f, x, eps=None):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    g, h = np.zeros(n), np.zeros(n)
    for i in range(n):
        h[i] = eps
        g[i] = (f(x+h) - f(x)) / eps
        h[i] = 0
    return g

def so_fc_fd(f, x, eps=None):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    H, hi, hj = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    eps2 = eps**2
    for i in range(n):
        hi[i] = eps
        for j in range(i+1):
            hj[j] = eps
            H[i, j] = (f(x+hi+hj) - f(x+hi) - f(x+hj) + f(x)) / eps2
            H[j, i] = H[i, j]
            hj[j] = 0  
        hi[i] = 0
    return H

def so_gc_fd(g, x, eps=None):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    H, h = np.zeros((n, n)), np.zeros(n)
    gx, gxh = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        h[i] = eps
        gx[i] = g(x)
        gxh[i] = g(x+h)
        h[i] = 0
    for i in range(n):
        for j in range(i+1):
            H[i, j] = ((gxh[i, j] - gx[i, j]) + (gxh[j, i] - gx[j, i])) / (2 * eps)
            H[j, i] = H[i, j]
    return H

def fo_fc_cd(f, x, eps=None):
    if eps is None:
        eps = (np.finfo(float).eps)**(1.0/3.0)
    n = len(np.asarray(x))
    g, h = np.zeros(n), np.zeros(n)
    for i in range(n):
        h[i] = eps
        g[i] = (f(x+h) - f(x - h)) / (2 * eps)
        h[i] = 0
    return g


def so_fc_cd(f, x, *args, eps=None):
    p = len(np.asarray(x))
    if eps is None:
        eps = (np.finfo(float).eps)**(1./3.)
    H = np.zeros((p, p))
    ei = np.zeros(p)
    ej = np.zeros(p)
    for i in range(p):
        for j in range(i+1):
            ei[i], ej[j] = eps, eps
            if i==j:
                dn = -f(x+2*ei)+16*f(x+ei)-30*f(x)+16*f(x-ei)-f(x-2*ei)
                nm = 12*eps**2
                H[i, j] = dn/nm  
            else:
                dn = f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej)
                nm = 4*eps*eps
                H[i, j] = dn/nm  
                H[j, i] = dn/nm  
            ei[i], ej[j] = 0.0, 0.0
    return H
        
def so_gc_cd(g, x, *args, eps=None):
    if eps is None:
        eps = (np.finfo(float).eps)**(1./3.)
    n = len(np.asarray(x))
    H, h = np.zeros((n, n)), np.zeros(n)
    gxp, gxn = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        h[i] = eps
        gxp[i] = g(x+h)
        gxn[i] = g(x-h)
        h[i] = 0
    for i in range(n):
        for j in range(i+1):
            H[i, j] = ((gxp[i, j] - gxn[i, j] + gxp[j, i] - gxn[j, i])) / (4 * eps)
            H[j, i] = H[i, j]
    return H


def fd_coefficients(points, order):
    A = np.zeros((len(points), len(points)))
    A[0] = 1
    for i in range(len(points)):
        A[i] = np.asarray(points)**(i)
    b = np.zeros(len(points))
    b[order] = sp.special.factorial(order)
    c = np.linalg.inv(A).dot(b)
    return c
        
    
def finite_diff(f, x, epsilon=None, order=1, points=None):
    if points is None:
        points = np.arange(-4, 5)
    if epsilon is None:
        epsilon = (np.finfo(float).eps)**(1./3.)
    coefs = fd_coefficients(points, order)
    df = 0.0
    for c, p in list(zip(coefs, points)):
        df+=c*f(x+epsilon*p)
    df = df / (epsilon**order)
    return df
        
    
    
    
    