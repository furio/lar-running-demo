# -*- coding: utf-8 -*-
"""
The MIT License
===============
    
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import collections
import scipy.sparse
import numpy as np
from scipy import zeros,arange,mat,amin,amax
from scipy.sparse import vstack,hstack,csr_matrix,coo_matrix,lil_matrix,triu
from scipy.spatial import Delaunay
from scipy.linalg import *
from matrixutil_no_accel import *
import time as tm

try:
    xrange
except NameError:
    xrange = range

# ------------------------------------------------------------
# Logging & Timer 
# ------------------------------------------------------------

logging_level = 0; 

# 0 = no_logging
# 1 = few details
# 2 = many details
# 3 = many many details

def log(n, l):
	for s in l:
		print("Log:", s)

timer = 1

timer_last =  tm.time()

def timer_start(s):
	global timer_last;
	if __name__=="__main__" and timer == 1:   
		log(3, ["Timer start:" + s])
	timer_last = tm.time()

def timer_stop():
	global timer_last
	if __name__=="__main__" and timer == 1:   
		log(3, ["Timer stop :" + str(tm.time() - timer_last)])

# ------------------------------------------------------------

self_test=False

#------------------------------------------------------------------
#--geometry layer (using PyPlasm)----------------------------------
#------------------------------------------------------------------
# AA
def AA(f):
    def AA0(args): return map(f, args)
    return AA0

def TRANS(List):
  return np.array(List).transpose().toList()

###################################################################

#------------------------------------------------------------------
#-- basic LAR software layer --------------------------------------
#------------------------------------------------------------------

#--coo is the standard rep using non-ordered triples of numbers----
#--coo := (row::integer, column::integer, value::float)------------


#------------------------------------------------------------------
def cooCreateFromBrc(ListOfListOfInt):
    COOm = [[k,col,1] for k,row in enumerate(ListOfListOfInt)
                   for col in row ]
    return COOm


#------------------------------------------------------------------
def csrCreateFromCoo(COOm):
    CSRm = format(COOm,"csr")
    return CSRm

#------------------------------------------------------------------
def csrCreate(BRCm,shape=(0,0)):
    if shape == (0,0):
        out = csrCreateFromCoo(cooCreateFromBrc(BRCm))
        return out
    else:
        CSRm = scipy.sparse.csr_matrix(shape)
        for i,j,v in cooCreateFromBrc(BRCm):
            CSRm[i,j] = v
        return CSRm

#------------------------------------------------------------------
def csrGetNumberOfRows(CSRm):
    Int = CSRm.shape[0]
    return Int

#------------------------------------------------------------------
def csrGetNumberOfColumns(CSRm):
    Int = CSRm.shape[1]
    return Int

#------------------------------------------------------------------
def csrToMatrixRepresentation(CSRm):
    nrows = csrGetNumberOfRows(CSRm)
    ncolumns = csrGetNumberOfColumns(CSRm)
    ScipyMat = zeros((nrows,ncolumns),int)
    C = CSRm.tocoo()
    for triple in zip(C.row,C.col,C.data):
        ScipyMat[triple[0],triple[1]] = triple[2]
    return ScipyMat

#------------------------------------------------------------------
def csrToBrc(CSRm):
    nrows = csrGetNumberOfRows(CSRm)
    C = CSRm.tocoo()
    out = [[] for i in xrange (nrows)]
    [out[i].append(j) for i,j in zip(C.row,C.col)]
    return out

#------------------------------------------------------------------
#--matrix utility layer--------------------------------------------
#------------------------------------------------------------------

#------------------------------------------------------------------
def csrIsA(CSRm):
    test = CSRm.check_format(True)
    return test==None

#------------------------------------------------------------------
def csrGet(CSRm,row,column):
    Num = CSRm[row,column]
    return Num

#------------------------------------------------------------------
def csrSet(CSRm,row,column,value):
    CSRm[row,column] = value
    return None

#------------------------------------------------------------------
def csrAppendByRow(CSRm1,CSRm2):
    CSRm = vstack([CSRm1,CSRm2])
    return CSRm

#------------------------------------------------------------------
def csrAppendByColumn(CSRm1,CSRm2):
    CSRm = hstack([CSRm1,CSRm2])
    return CSRm

#------------------------------------------------------------------
def csrSplitByRow(CSRm,k):
    CSRm1 = CSRm[:k]
    CSRm2 = CSRm[k:]
    return CSRm1,CSRm2

#------------------------------------------------------------------
def csrSplitByColumn(CSRm,k):
    CSRm1 = CSRm.T[:k]
    CSRm2 = CSRm.T[k:]
    return CSRm1.T,CSRm2.T

#------------------------------------------------------------------
#--sparse matrix operations layer----------------------------------
#------------------------------------------------------------------

def csrBoundaryFilter(CSRm, facetLengths):
	maxs = [max(CSRm[k].data) for k in xrange(CSRm.shape[0])]
	inputShape = CSRm.shape

	coo = CSRm.tocoo()

	row = [] # np.array([]).astype(np.int32);
	col = [] # np.array([]).astype(np.int32);
    # data = [] # np.array([]).astype(np.int32);

	k = 0
	while (k < len(coo.data)):      
		if coo.data[k] == maxs[coo.row[k]]:
			row.append(coo.row[k])
			col.append(coo.col[k])
		k += 1
    
	data = np.ones(len(col),dtype=np.int32)
	mtx = coo_matrix( (data, ( np.array(row).astype(np.int32), np.array(col).astype(np.int32) )), shape=inputShape)

	out = mtx.tocsr()
	return out

#------------------------------------------------------------------
def csrBinFilter(CSRm):
    # can be done in parallel (by rows)
	inputShape = CSRm.shape
	coo = CSRm.tocoo()
    
	k = 0
	while (k < len(coo.data)):
		if (coo.data[k] % 2 == 1): 
			coo.data[k] = 1
		else: 
			coo.data[k] = 0
		k += 1
    #mtx = coo_matrix((coo.data, (coo.row, coo.col)), shape=inputShape)
    #out = mtx.tocsr()
    #return out
	return coo.tocsr()

#------------------------------------------------------------------
def csrPredFilter(CSRm, pred):
    # can be done in parallel (by rows)
    coo = CSRm.tocoo()
    triples = [[row,col,val] for row,col,val in zip(coo.row,coo.col,coo.data)
               if pred(val)]
    i, j, data = TRANS(triples)
    CSRm = scipy.sparse.coo_matrix((data,(i,j)),CSRm.shape).tocsr()
    return CSRm

#------------------------------------------------------------------
#--topology interface layer----------------------------------------
#------------------------------------------------------------------

#------------------------------------------------------------------
def csrCreateTotalChain(kn):
    csrMat = csrCreateFromCoo(cooCreateFromBrc(TRANS([kn*[0]])))
    return csrMat

#------------------------------------------------------------------
def csrCreateUnitChain(kn,k):
    CSRout = lil_matrix((kn, 1))
    CSRout[k,0] = 1
    return CSRout.tocsr()

#------------------------------------------------------------------
def csrExtractAllGenerators(CSRm):
    listOfListOfNumerals = [csrTranspose(CSRm)[k].tocoo().col.tolist()
                            for k in xrange(CSRm.shape[1])]
    return listOfListOfNumerals

#------------------------------------------------------------------
def csrChainToCellList(CSRm):
    coo = CSRm.tocoo()
    ListOfInt = [theRow for k,theRow in enumerate(coo.row) if coo.data[k]==1]
    return ListOfInt

#------------------------------------------------------------------
#--topology query layer--------------------------------------------
#------------------------------------------------------------------

#------------------------------------------------------------------
def larCellAdjacencies(CSRm):
    CSRm = matrixProduct(CSRm,csrTranspose(CSRm))
    return CSRm

#------------------------------------------------------------------
def larCellIncidences(CSRm1,CSRm2):
    return matrixProduct(CSRm1, csrTranspose(CSRm2))

#------------------------------------------------------------------
# FV = d-chain;  EV = (d-1)-chain

def larBoundary(EV,FV):
    e = len(EV)
    f = len(FV)
    v = max(AA(max)(FV))+1
    #v = FV[-1][-1]+1  # at least with images ...
    csrFV = csrCreate(FV)#,shape=(f,v))
    csrEV = csrCreate(EV)#,shape=(e,v))
    facetLengths = [csrCell.getnnz() for csrCell in csrEV]
    temp = larCellIncidences(csrEV,csrFV)
    csrBoundary_2 = csrBoundaryFilter(temp,facetLengths)
    return csrBoundary_2

#------------------------------------------------------------------
def larBoundaryChain(csrBoundaryMat,brcCellList):
    n,m = csrBoundaryMat.shape
    data = scipy.ones(len(brcCellList))
    i = brcCellList
    j = scipy.zeros(len(brcCellList))
    csrChain = coo_matrix((data,(i,j)),shape=(m,1)).tocsr()
    csrmat = matrixProduct(csrBoundaryMat,csrChain)
    out = csrBinFilter(csrmat)
    return out

#------------------------------------------------------------------
def larCoboundaryChain(csrCoBoundaryMat,brcCellList):
    m = csrGetNumberOfColumns(csrCoBoundaryMat)
    csrChain = sum([csrCreateUnitChain(m,k) for k in brcCellList])
    return csrBinFilter(matrixProduct(csrCoBoundaryMat,csrChain))

#------------------------------------------------------------------
#--model geometry layer--------------------------------------------
#--larOp : model -> model------------------------------------------
#------------------------------------------------------------------
# model = (vertices, topology)
#------------------------------------------------------------------
# binary product of cell complexes

def larProduct(models):
    model1,model2 = models
    V, cells1 = model1
    W, cells2 = model2
    verts = collections.OrderedDict(); k = 0
    for v in V:
        for w in W:
            vertex = tuple(v+w)
            if not verts.has_key(vertex):
                verts[vertex] = k
                k += 1
    cells = [ sorted([verts[tuple(V[v]+W[w])] for v in c1 for w in c2])
             for c1 in cells1 for c2 in cells2]

    model = AA(list)(verts.keys()), sorted(cells)
    return model

#------------------------------------------------------------------
# extrusion of simplicial complexes
# combinatorial algorithm

def cumsum(iterable):
    # cumulative addition: list(cumsum(range(4))) => [0, 1, 3, 6]
    iterable = iter(iterable)
    s = iterable.next()
    yield s
    for c in iterable:
        s = s + c
        yield s