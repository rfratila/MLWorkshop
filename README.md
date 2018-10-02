# MLWorkshop
Used for the Machine Learning Workshop

## Dependencies:
```
Numpy : '1.12.0'
Scipy: '0.18.1'
Matplotlib: '1.5.3'
Theano: '0.9.0dev4'
Lasagne: '0.2.dev1'
```
## Running the script
Open DeepEEG.py in your IDE of choice and modify the following lines:
```
#MODIFY THESE
	#*******************************************
	trainTime = 10.0/60 #in hours
	modelName='whatever_you_want'
	#*******************************************
```

Then to run the training script run:
```
python DeepEEG.py
```

Once training has completed, to visualize the training run:
```
python viewMetrics.py whatever_you_wantstats.pickle
```
