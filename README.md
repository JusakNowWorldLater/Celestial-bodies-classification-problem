**Guide:**
To use test program you should run in terminal such command:
python3 main.py [path_to_train] [path_to_unlabeled_data] [path_to_test] [path_to_predictions]

In example:
python3 main.py data/train.csv data/unlabeled.csv data/test.csv results/prediction.csv

**Dependencies:**
To run this script("main.py") you should have installed following libraries: numpy, pandas, sklearn. 
To run "train_predict_model.py" you should have: pandas, numpy, os, sklearn, keras, tensorflow
To see research "research.ipynb" you should have: numpy, pandas, sklearn, matplotlib, plotnine, seaborn

**Results:**
* Random forest classifier:
	* Accuracy: 0.869
	* Macro-avg f1: 0.840
* Simple NN:
	* Accuracy: 0.839
	* Macro-avg f1: 0.806
	* Loss: 0.449

Remark: *All results measuared on test data*