#Import Statements
import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse import csc_matrix
import math
import matplotlib as mpl
import imageio
import os
import networkx as nx
import csv
import warnings

#Define variables
nblocks = 5
n = 10000

#p = intrablock connection probability, q = interblock connection probability
p = 0.1
q = 0.001

#Parameters
tau = 0.01
T = 3
days = np.zeros(n)

#Initialize
i = np.zeros(n)
#Initialize infected individual
i[0] = 1
s = 1 - i
r = np.zeros(n)

#Arrays that store information
iSums = []
sSums= []
rSums = []
sSums.append(np.sum(s))
iSums.append(np.sum(i))
rSums.append(np.sum(r))

#Define the model

#############################################################
## FOR NOW, VERTICES ARE EVENLY DISTRIBUTED BETWEEN BLOCKS ##
#############################################################
sizes = [n//nblocks,n//nblocks, n//nblocks,n//nblocks, n//nblocks]
probs = [[p,q,q,q,q],[q,p,q,q,q],[q,q,p,q,q],[q,q,q,p,q],[q,q,q,q,p]]
g = nx.stochastic_block_model(sizes, probs)
W = nx.adjacency_matrix(g)
def makeModel(numpts = 1000, numblocks = 5, intra=0.1, inter = 0.001, tauVal = 0.01, TVal = 3):
  global nblocks, n, i, s, r, iSums, rSums, sSums, p, q, sizes, probl, g, W, tau, T, days
  #Define variables
  nblocks = numblocks
  n = numpts
  #Initialize
  i = np.zeros(n)
  #Initialize infected individual
  i[0] = 1
  s = 1 - i
  r = np.zeros(n)

  #Arrays that store information
  iSums = []
  sSums= []
  rSums = []
  sSums.append(np.sum(s))
  iSums.append(np.sum(i))
  rSums.append(np.sum(r))
  #p = intrablock connection probability, q = interblock connection probability
  p = intra
  q = inter
  #Parameters
  tau = tauVal
  T = TVal
  days = np.zeros(n)
  sizes = []
  for k in range(nblocks):
    sizes.append(n//nblocks)
  probs = []
  for k1 in range(nblocks):
    dummy = []
    for k2 in range(nblocks):
      if k1 == k2:
        dummy.append(p)
      else:
        dummy.append(q)
    probs.append(dummy)
  g = nx.stochastic_block_model(sizes, probs)
  W = nx.adjacency_matrix(g)


#This is the code to display an image of the points on the graph, along with their color
def step():
  global n,i,s,r, W, days
  #Step 2(a)
  #W = csc_matrix(W).toarray() #converting the sparse array to np array because multiplication simplicity :)
  I = W.dot(i)
  q = np.zeros(n)
  #Step 2(b)
  for k in range(n):
    q[k] = math.exp((-1) * tau * I[k])
  #q = math.exp((-1) * tau * I)
  x = np.random.rand(n)
  z = np.zeros(n)
  for k in range(n):
    if x[k]>q[k]:
      z[k] = 1
  #Step 2(c)
  #for k in range(n):
  #  i[k] = (1 - s[k])*i[k] + s[k] * z[k]
  i = (1-s) * i + s * z
  #Step 2(d)
  #for k in range(n):
    #s[k] = (1-i[k]) * (1-r[k])
  s = (1-i) * (1-r)
  #Step 2(e)
  #for k in range(n):
   # days[k] = days[k] + i[k]
  days = days + i
  #Step 2(f)
  for k in range(n):
    if days[k] > T:
      r[k] = 1
      i[k] = 0
  #Present graph
  #COVIDimage()
  #Record information
  return np.sum(s), np.sum(i), np.sum(r)

def simulate(reps = -1):
  m=0
  if reps == -1:
    while m == 0 or (i != 0).any():
      a,b,c = step()
      sSums.append(a)
      iSums.append(b)
      rSums.append(c)
      m = m + 1
    #print('Done')
    return m
  else:
    for k in range(reps):
      a,b,c = step()
      sSums.append(a)
      iSums.append(b)
      rSums.append(c)
      m = m + 1

#Find the data needed to train the NN
def findResults():
  #Find the max infected
  maximum = max(iSums)
  #Find the first 100 eigenvalues
  G = gl.graph(W)
  vals, vecs = G.eigen_decomp(normalization = 'normalized', k = 100)
  return maximum, vals