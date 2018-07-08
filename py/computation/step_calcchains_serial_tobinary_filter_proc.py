# -*- coding: utf-8 -*-

from lar import invertIndex, indFun
from scipy.sparse import vstack,hstack,csr_matrix,coo_matrix
import json
import scipy
import numpy as np
import time as tm
import gc
from pngstack2array3d import *
import struct
import getopt, sys
import traceback
#
import matplotlib.pyplot as plt
# threading
import asyncio
# pycuda
import pyculib as p

try:
    xrange
except NameError:
    xrange = range

# ------------------------------------------------------------
# Logging & Timer 
# ------------------------------------------------------------

def log(n, l):
	for s in l:
		print( "Log:", s)

# ------------------------------------------------------------
# Configuration parameters
# ------------------------------------------------------------

PNG_EXTENSION = ".png"
BIN_EXTENSION = ".bin"

# ------------------------------------------------------------
# Utility toolbox
# ------------------------------------------------------------

def countFilesInADir(directory):
	fList = [name for name in os.listdir(directory) if os.path.isfile(directory + '/' + name) and not name.startswith('.') ]
	return len(fList)
	
def isArrayEmpty(arr):
	return all(e == 0 for e in arr)
	
# ------------------------------------------------------------
def writeOffsetToFile(file, offsetCurr):
	file.write( struct.pack('>I', offsetCurr[0]) )
	file.write( struct.pack('>I', offsetCurr[1]) )
	file.write( struct.pack('>I', offsetCurr[2]) )
# ------------------------------------------------------------

async def computeChainsAsync(queueIn, queueOut, bordo3):
	# (False/True, {x:,y:,z:}, csrMv)
	
	log(2, [ 'computeChainsAsync' ])

	# init pycuda
	sparseIstance = p.sparse.Sparse()
	cudaCSR1 = p.sparse.csr_matrix((bordo3.data, bordo3.indices, bordo3.indptr), shape=bordo3.shape)

	while True:
		end,coords,chain = await queueIn.get()
		if end is False:
			queueIn.task_done()
			break
		
		# larBoundaryChain
		log(2, [ 'computeChainsAsync - Task' ])
		cudaCSR2 = p.sparse.csr_matrix((chain.data, chain.indices, chain.indptr), shape=chain.shape)
		cudaCSRm = sparseIstance.csrgemm_ez(cudaCSR1, cudaCSR2)

		hostM = csr_matrix((cudaCSRm.data, cudaCSRm.indices, cudaCSRm.indptr), shape=cudaCSRm.shape, dtype=np.int32).tocoo()

		newRow = []
		k = 0
		while (k < len(hostM.data) - 1):
			if (hostM.data[k] % 2 == 1):
				newRow.append(hostM.row[k])

			k += 1

		hostCSRm = coo_matrix( (np.ones(len(newRow),dtype=np.int8), (np.array(newRow).astype(np.int32), np.zeros(len(newRow),dtype=np.int8))), shape=cudaCSRm.shape ).tocsr()

		await queueOut.put( ( True, coords, hostCSRm ) )
		queueIn.task_done()
	
	await queueOut.put( (False, None, None) )

def addrChain(x, y, z, nx, ny, nz):
	return x + (nx) * (y + (ny) * (z))

async def createChainsAsync(queueIn, queueOut, imageHeight,imageWidth, imageDx,imageDy,imageDz, colors,pixelCalc,centroidsCalc, colorIdx, imageDir, bordo3shape):
	# (False/True, (startImage,endImage))

	log(2, [ 'createChainsAsync' ])

	xEnd, yEnd = 0,0
	beginImageStack = 0
	saveTheColors = centroidsCalc
	saveTheColors = np.array( sorted(saveTheColors.reshape(1,colors)[0]), dtype=np.int )

	while True:
		end,startEnd = await queueIn.get()
		if end is False:
			queueIn.task_done()
			break

		try:
			startImage, endImage = startEnd
			log(2, [ "Working task: " +str(startImage) + "-" + str(endImage) + " [loading colors]" ])
			theImage,colors,theColors = pngstack2array3d(imageDir, startImage, endImage, colors, pixelCalc, centroidsCalc)
			log(2, [ "Working task: " +str(startImage) + "-" + str(endImage) + " [comp loop]" ])
			for xBlock in xrange(imageHeight//imageDx):
				# print "Working task: " +str(startImage) + "-" + str(endImage) + " [Xblock]"
				for yBlock in xrange(imageWidth//imageDy):
					# print "Working task: " +str(startImage) + "-" + str(endImage) + " [Yblock]"
					xStart, yStart = xBlock * imageDx, yBlock * imageDy
					xEnd, yEnd = xStart+imageDx, yStart+imageDy
								
					image = theImage[:, xStart:xEnd, yStart:yEnd]
					nz,nx,ny = image.shape

					chains3D = []
					zStart = startImage - beginImageStack

					for x in xrange(nx):
						for y in xrange(ny):
							for z in xrange(nz):
								if (image[z,x,y] == saveTheColors[colorIdx]):
									chains3D.append(addrChain(x,y,z,nx,ny,nz))
					
					csrChains3D = coo_matrix((scipy.ones(len(chains3D)),(chains3D,scipy.zeros(len(chains3D)))),shape=(bordo3shape[1],1),dtype=np.float32).tocsr()
					await queueOut.put( (True, {"z": zStart, "x": xStart, "y": yStart}, csrChains3D) )

			queueIn.task_done()
		except:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
			log(1, [ "Error: " + ''.join('!! ' + line for line in lines) ])  # Log it or whatever here			
	
	await queueOut.put( (False, None, None) )

async def startChainProcessAsync(queueOut, imageDepth, imageDz):
	log(2, [ 'startChainProcessAsync ' + str(imageDepth) + '-' + str(imageDz) ])
	endImage = 0

	for j in xrange(imageDepth//imageDz):
		startImage = endImage
		endImage = startImage + imageDz
		log(2, [ 'startChainProcessAsync', 'put', str((startImage, endImage)) ])
		await queueOut.put( (True, (startImage, endImage)) )
		log(2, [ "Added task: " + str(j) + " -- (" + str(startImage) + "," + str(endImage) + ")" ])
	
	await queueOut.put((False, None))

def startComputeChainsCuda(imageHeight,imageWidth,imageDepth, imageDx,imageDy,imageDz, BORDER_FILE, colors,pixelCalc,centroidsCalc, colorIdx,INPUT_DIR,DIR_O):
	returnValue = 2
	
	beginImageStack = 0
	endImage = beginImageStack
	
	saveTheColors = centroidsCalc
	log(2, [ centroidsCalc ])
	saveTheColors = np.array( sorted(saveTheColors.reshape(1,colors)[0]), dtype=np.int )
	log(2, [ saveTheColors ])

	bordo3 = None
	with open(BORDER_FILE, "r") as file:
		bordo3_json = json.load(file)
		ROWCOUNT = bordo3_json['ROWCOUNT']
		COLCOUNT = bordo3_json['COLCOUNT']
		ROW = np.asarray(bordo3_json['ROW'], dtype=np.int32)
		COL = np.asarray(bordo3_json['COL'], dtype=np.int32)
		DATA = np.asarray(bordo3_json['DATA'], dtype=np.float32)
		bordo3 = csr_matrix((DATA,COL,ROW),shape=(ROWCOUNT,COLCOUNT))

	try:
		log(2, [ 'Start Cuda' ])
		
		loop = asyncio.get_event_loop()
		queueStackImages = asyncio.Queue(loop=loop)
		queueChainList = asyncio.Queue(loop=loop)
		queueExtractedBorders = asyncio.Queue(loop=loop)

		log(2, [ "Waiting for completion..." ])
		loop.run_until_complete(asyncio.gather(
			startChainProcessAsync(queueStackImages, imageDepth, imageDz), 
			createChainsAsync(queueStackImages, queueChainList, imageHeight,imageWidth,imageDx,imageDy,imageDz, colors,pixelCalc,centroidsCalc, colorIdx, INPUT_DIR, bordo3.shape), 
			computeChainsAsync(queueChainList, queueExtractedBorders, bordo3),
			loop=loop))

		try:
			log(2, [ "Dumping chains..." ])
			currResultChain = queueExtractedBorders.get_nowait()
			while currResultChain[0] is True:
				serializeChains(currResultChain[1], currResultChain[2], DIR_O)
				currResultChain = queueExtractedBorders.get_nowait()
		except:
			pass

		loop.close()

		log(1, [ "Completed "])
		returnValue = 0
	except:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		log(1, [ "Error: " + ''.join('!! ' + line for line in lines) ])  # Log it or whatever here

	return returnValue

def serializeChains(deltaCoords, chain, DIR_O):
	ROWCOUNT = chain.shape[0]
	COLCOUNT = chain.shape[1]
	ROW = chain.indptr.tolist()
	COL = chain.indices.tolist()
	DATA = chain.data.tolist()

	fileName = DIR_O + "/" + str(deltaCoords["z"]) + "-" + str(deltaCoords["y"]) + "-" + str(deltaCoords["x"]) + "-chain.json" 

	with open(fileName, "w") as file:
		json.dump({"DELTA":deltaCoords, "CHAIN": {"ROWCOUNT":ROWCOUNT, "COLCOUNT":COLCOUNT, "ROW":ROW, "COL":COL, "DATA":DATA }}, file, separators=(',',':'))
		file.flush()

def runComputation(imageDx,imageDy,imageDz, colors,coloridx, V,FV, INPUT_DIR,BEST_IMAGE,BORDER_FILE,DIR_O):
	imageHeight,imageWidth = getImageData(INPUT_DIR+str(BEST_IMAGE)+PNG_EXTENSION)
	imageDepth = countFilesInADir(INPUT_DIR)
	Nx,Ny,Nz = imageHeight//imageDx, imageWidth//imageDx, imageDepth//imageDz
	returnValue = 2
	
	try:
		pixelCalc, centroidsCalc = centroidcalc(INPUT_DIR, BEST_IMAGE, colors)
		returnValue = startComputeChainsCuda(imageHeight,imageWidth,imageDepth, imageDx,imageDy,imageDz, BORDER_FILE, colors,pixelCalc,centroidsCalc, coloridx,INPUT_DIR,DIR_O)

	except:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		log(1, [ "Error: " + ''.join('!! ' + line for line in lines) ])  # Log it or whatever here
		returnValue = 2
		
	sys.exit(returnValue)

# python py/computation/step_calcchains_serial_tobinary_filter_proc.py -b $(pwd)/testBorder/bordo3_8-8-8.json -x 8 -y 8 -z 8 -i $(pwd)/testImages -c 2 -d 1 -q 0 -o $(pwd)/testOutchains
def main(argv):
	ARGS_STRING = 'Args: -b <borderfile> -x <borderX> -y <borderY> -z <borderZ> -i <inputdirectory> -c <colors> -d <coloridx> -o <outputdir> -q <bestimage>'

	try:
		opts, args = getopt.getopt(argv,"rb:x:y:z:i:c:d:o:q:")
	except getopt.GetoptError:
		print(ARGS_STRING)
		sys.exit(2)
	
	nx = ny = nz = imageDx = imageDy = imageDz = 64
	colors = 2
	coloridx = 0
	
	mandatory = 6
	#Files
	BORDER_FILE = 'bordo3.json'
	BEST_IMAGE = ''
	DIR_IN = ''
	DIR_O = ''
	
	for opt, arg in opts:
		if opt == '-x':
			nx = ny = nz = imageDx = imageDy = imageDz = int(arg)
			mandatory = mandatory - 1
		elif opt == '-y':
			ny = nz = imageDy = imageDz = int(arg)
		elif opt == '-z':
			nz = imageDz = int(arg)
		elif opt == '-i':
			DIR_IN = arg + '/'
			mandatory = mandatory - 1
		elif opt == '-b':
			BORDER_FILE = arg
			mandatory = mandatory - 1
		elif opt == '-o':
			mandatory = mandatory - 1
			DIR_O = arg
		elif opt == '-c':
			mandatory = mandatory - 1
			colors = int(arg)
		elif opt == '-d':
			mandatory = mandatory - 1
			coloridx = int(arg)			
		elif opt == '-q':
			BEST_IMAGE = int(arg)		
			
	if mandatory != 0:
		print('Not all arguments where given')
		print(ARGS_STRING)
		sys.exit(2)
		
	if (coloridx >= colors):
		print('Not all arguments where given (coloridx >= colors)')
		print(RGS_STRING)
		sys.exit(2)
	
	chunksize = nx * ny + nx * nz + ny * nz + 3 * nx * ny * nz
	V = [[x,y,z] for z in xrange(nz+1) for y in xrange(ny+1) for x in xrange(nx+1) ]
	
	v2coords = invertIndex(nx,ny,nz)
	ind = indFun(nx,ny,nz)

	FV = []
	for h in xrange(len(V)):
		x,y,z = v2coords(h)
		if (x < nx) and (y < ny): FV.append([h,ind(x+1,y,z),ind(x,y+1,z),ind(x+1,y+1,z)])
		if (x < nx) and (z < nz): FV.append([h,ind(x+1,y,z),ind(x,y,z+1),ind(x+1,y,z+1)])
		if (y < ny) and (z < nz): FV.append([h,ind(x,y+1,z),ind(x,y,z+1),ind(x,y+1,z+1)])

	runComputation(imageDx, imageDy, imageDz, colors, coloridx, V, FV, DIR_IN, BEST_IMAGE, BORDER_FILE, DIR_O)

if __name__ == "__main__":
	main(sys.argv[1:])