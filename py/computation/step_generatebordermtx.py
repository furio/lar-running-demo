# -*- coding: utf-8 -*-

from lar import larBoundaryNew, invertIndex, indFun
import json
import scipy
import numpy as np
import time as tm
import gc
import struct
import sys
import getopt, sys
import traceback

try:
    xrange
except NameError:
    xrange = range

logging_level = 0

def log(n, l):
	for s in l:
		print("Log:", s)

# ------------------------------------------------------------
# Computation of ∂3 operator on the image space
# ------------------------------------------------------------

def computeBordo3(FV,CV,Vlen,FVcount,CVcount,inputFile='bordo3.json'):
	log(1, ["bordo3 = Starting"])
	bordo3 = larBoundaryNew(FV,(FVcount,Vlen),CV,(CVcount,Vlen))
	log(3, ["bordo3 = " + str(bordo3.shape)])
	log(1, ["bordo3 = Done"])

	ROWCOUNT = bordo3.shape[0]
	COLCOUNT = bordo3.shape[1]
	ROW = bordo3.indptr.tolist()
	COL = bordo3.indices.tolist()
	DATA = bordo3.data.tolist()

	with open(inputFile, "w") as file:
		json.dump({"ROWCOUNT":ROWCOUNT, "COLCOUNT":COLCOUNT, "ROW":ROW, "COL":COL, "DATA":DATA }, file, separators=(',',':'))
		file.flush()

def main(argv):
	ARGS_STRING = 'Args: -x <borderX> -y <borderY> -z <borderZ> -o <outputdir>'

	try:
		opts, args = getopt.getopt(argv,"o:x:y:z:")
	except getopt.GetoptError:
		print(ARGS_STRING)
		sys.exit(2)
	
	mandatory = 2
	#Files
	DIR_OUT = ''
	
	for opt, arg in opts:
		if opt == '-x':
			nx = ny = nz = int(arg)
			mandatory = mandatory - 1
		elif opt == '-y':
			ny = nz = int(arg)
		elif opt == '-z':
			nz = int(arg)
		elif opt == '-o':
			DIR_OUT = arg
			mandatory = mandatory - 1
			
	if mandatory != 0:
		print('Not all arguments where given')
		print(ARGS_STRING)
		sys.exit(2)
		
	log(1, ["nx, ny, nz = " + str(nx) + "," + str(ny) + "," + str(nz)])

	ind = indFun(nx,ny,nz)

	def the3Dcell(coords):
		x,y,z = coords
		return [ind(x,y,z),ind(x+1,y,z),ind(x,y+1,z),ind(x,y,z+1),ind(x+1,y+1,z),
				ind(x+1,y,z+1),ind(x,y+1,z+1),ind(x+1,y+1,z+1)]	
	
	# Construction of vertex coordinates (nx * ny * nz)
	# ------------------------------------------------------------

	#V = list()
	Vlen = (nx+1) * (ny+1) * (nz+1)
	
	log(3, ["Vlen = " + str(Vlen)])

	# V = [[x,y,z] for z in xrange(nz+1) for y in xrange(ny+1) for x in xrange(nx+1) ]

	# log(3, ["V = " + str(V)])

	# Construction of CV relation (nx * ny * nz)
	# ------------------------------------------------------------
	CV = list()
	CVcount = 0

	for z in xrange(nz):
		for y in xrange(ny):
			for x in xrange(nx):
				for idx in the3Dcell([x,y,z]):
					CV.append((CVcount,idx,1))

				CVcount = CVcount + 1

	# CV = [the3Dcell([x,y,z]) for z in xrange(nz) for y in xrange(ny) for x in xrange(nx)]

	log(3, ["len(CV) = " + str(len(CV))])
	log(3, ["CVcount = " + str(CVcount)])

	# Construction of FV relation (nx * ny * nz)
	# ------------------------------------------------------------

	FV = list()
	FVcount = 0

	v2coords = invertIndex(nx,ny,nz)

	for h in xrange(Vlen):
		x,y,z = v2coords(h)
		faceArray = []
		if (x < nx) and (y < ny): 
			faceArray = [h,ind(x+1,y,z),ind(x,y+1,z),ind(x+1,y+1,z)]
		if (x < nx) and (z < nz): 
			faceArray = [h,ind(x+1,y,z),ind(x,y,z+1),ind(x+1,y,z+1)]
		if (y < ny) and (z < nz): 
			faceArray = [h,ind(x,y+1,z),ind(x,y,z+1),ind(x,y+1,z+1)]

		if len(faceArray) > 0:
			for idx in faceArray:
				FV.append((FVcount,idx,1))
			FVcount = FVcount + 1

	log(3, ["len(FV) = " + str(len(FV))])
	log(3, ["FVcount = " + str(FVcount)])
	
	fileName = DIR_OUT+'/bordo3_'+str(nx)+'-'+str(ny)+'-'+str(nz)+'.json'
	
	try:
		computeBordo3(FV,CV,Vlen,FVcount,CVcount,fileName)
	except:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		log(1, [ "Error: " + ''.join('!! ' + line for line in lines) ])  # Log it or whatever here
		sys.exit(2)

if __name__ == "__main__":
	main(sys.argv[1:])