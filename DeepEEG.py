import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T
import os

from collections import OrderedDict
import pylab #for graphing

import json
from random import shuffle

import time

#be able to read from an Attentive folder and create their truth values
def getJsonData(dataPath):
	if 'inattentive' in dataPath:
		trainOut = numpy.array([[0,1]]) #this will contain the actual state of the brain: inattentive
	else:
		trainOut = numpy.array([[1,0]]) #this will contain the actual state of the brain: attentive
	data =[]
	res = {}
	with open(dataPath) as infile:
		res = json.load(infile)
	for timeStamp in res['data']:
		data.append(numpy.array(timeStamp['channel_values'],dtype='float32'))		
	data = numpy.stack(data,axis=1)
	data = numpy.resize(data,(data.shape[0],131072))
	data = OrderedDict(input=numpy.array(data, dtype='float32'), truth=numpy.array(trainOut, dtype='float32'))
	return data

def createNetwork(dimensions, input_var):
	#dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer
	#We have to specify the input size because of the dense layer
	print ("Creating Network...")

	print ('Input Layer:')
	network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
	
	print '	',lasagne.layers.get_output_shape(network)
	print ('Hidden Layer:')

	network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(5,5), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)
	print ('Output Layer:')
	print '	',lasagne.layers.get_output_shape(network)


	return network

#---------------------------------For training------------------------------------------
def createTrainer(network,input_var,y):
	print ("Creating Trainer...")
	#output of network
	out = lasagne.layers.get_output(network)
	#get all parameters from network
	params = lasagne.layers.get_all_params(network, trainable=True)
	#calculate a loss function which has to be a scalar
	cost = T.nnet.categorical_crossentropy(out, y).mean()
	#calculate updates using ADAM optimization gradient descent
	updates = lasagne.updates.adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
	#theano function to compare brain to their masks with ADAM optimization
	train_function = theano.function([input_var, y], updates=updates) # omitted (, allow_input_downcast=True)
	return train_function

def createValidator(network, input_var, y):
	print ("Creating Validator...")
	#We will use this for validation
	testPrediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
	testLoss = lasagne.objectives.categorical_crossentropy(testPrediction,y).mean()   #check how much error in prediction
	testAcc = T.mean(T.eq(T.argmax(testPrediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction

	validateFn = theano.function([input_var, y], [testLoss, testAcc])	 #check for error and accuracy percentage
	return validateFn

def saveModel(network,saveLocation='',modelName='brain1'):

	networkName = '%s%s.npz'%(saveLocation,modelName)
	print ('Saving model as %s'%networkName)
	numpy.savez(networkName, *lasagne.layers.get_all_param_values(network))

def loadModel(network, model='brain1.npz'):

	with numpy.load(model) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))] #gets all param values
		lasagne.layers.set_all_param_values(network, param_values)		  #sets all param values
	return network

def validateNetwork(network,input_var,validationSet):
	print ('Validating the network')
	out = lasagne.layers.get_output(network)
	test_fn = theano.function([input_var],out)
	truePos=falsePos=trueNeg=falseNeg = 0
	for sample in validationSet:
		data = getJsonData(sample)
		trainIn = data['input'].reshape([1,1] + list(data['input'].shape))

		print ("Sample: %s"%sample)
		if test_fn(trainIn)[0,0] == 1:
			print "Prediction: Attentive" 
			if "inattentive" in sample:
				falsePos+=1
			else:
				truePos+=1

		else: 
			print "Prediction: Inattentive"
			if "inattentive" in sample:
				trueNeg+=1
			else:
				falseNeg+=1
	print ('Samples: %s | True positives: %s | False positives: %s | True negatives: %s | False negatives: %s'%(len(validationSet),truePos,falsePos,trueNeg,falseNeg))

def main():
	dataPath = 'data'
	testReserve = 0.2
	validationReserve = 0.2
	trainingReserve = 1-(testReserve+validationReserve)
	input_var = T.tensor4('input')
	y = T.dmatrix('truth')

	trainFromScratch = True
	epochs = 10
	samplesperEpoch = 10

	#MODIFY THESE
	#*******************************************
	trainTime = 10.0/60 #in hours
	modelName='whatever_you_want'
	#*******************************************

	dataSet = []

	for patient in [dataPath]:

		attentivePath = os.path.join(dataPath,'attentive')
		inattentivePath = os.path.join(dataPath,'inattentive')

		if os.path.exists(attentivePath) and os.path.exists(inattentivePath):
			dataSet += [os.path.join(attentivePath,i) for i in os.listdir(attentivePath)]
			dataSet += [os.path.join(inattentivePath,i) for i in os.listdir(inattentivePath)  if i.endswith('.json')]
			shuffle(dataSet)

	print ("%i samples found"%len(dataSet))
	#This reserves the correct amount of samples for training, testing and validating
	trainingSet = dataSet[:int(trainingReserve*len(dataSet))]
	testSet = dataSet[int(trainingReserve*len(dataSet)):-int(testReserve*len(dataSet))]
	validationSet = dataSet[int(testReserve*len(dataSet) + int(trainingReserve*len(dataSet))):]

	inputDim = getJsonData(trainingSet[0])

	
	networkDimensions = (1,1,inputDim['input'].shape[0],inputDim['input'].shape[1])
	network  = createNetwork(networkDimensions, input_var)
	trainer = createTrainer(network,input_var,y)

	validator = createValidator(network,input_var,y)

	if not trainFromScratch:
		print ('loading a previously trained model...\n')
		network = loadModel(network,'Emily2Layer300000.npz')


	#print ("Training for %s epochs with %s samples per epoch"%(epochs,samplesperEpoch))
	record = OrderedDict(epoch=[],error=[],accuracy=[])

	print ("Training for %s hour(s) with %s samples per epoch"%(trainTime,samplesperEpoch))
	epoch = 0
	startTime = time.time()
	timeElapsed = time.time() - startTime
	#for epoch in xrange(epochs):            #use for epoch training
	while timeElapsed/3600 < trainTime :     #use for time training
		epochTime = time.time()
		print ("--> Epoch: %d | Time left: %.2f hour(s)"%(epoch,trainTime-timeElapsed/3600))
		for i in xrange(samplesperEpoch):
			chooseRandomly = numpy.random.randint(len(trainingSet))
			data = getJsonData(trainingSet[chooseRandomly])
			trainIn = data['input'].reshape([1,1] + list(data['input'].shape))
			trainer(trainIn, data['truth'])

		chooseRandomly = numpy.random.randint(len(testSet))
		print ("Gathering data...%s"%testSet[chooseRandomly])
		data = getJsonData(testSet[chooseRandomly])
		trainIn = data['input'].reshape([1,1] + list(data['input'].shape))
		error, accuracy = validator(trainIn, data['truth'])			     #pass modified data through network
		record['error'].append(error)
		record['accuracy'].append(accuracy)
		record['epoch'].append(epoch)
		timeElapsed = time.time() - startTime
		epochTime = time.time() - epochTime
		print ("	error: %s and accuracy: %s in %.2fs\n"%(error,accuracy,epochTime))
		epoch+=1

	validateNetwork(network,input_var,validationSet)

	saveModel(network=network,modelName=modelName)
	#import pudb; pu.db
	#save metrics to pickle file to be opened later and displayed
	import pickle
	#data = {'data':record}
	with open('%sstats.pickle'%modelName,'w') as output:
		#import pudb; pu.db
		pickle.dump(record,output)
	
if __name__ == "__main__":
    main()