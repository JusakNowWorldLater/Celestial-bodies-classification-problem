import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from plotnine import *

import sys

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def split_data(db):
	db_wthno_na = db.copy()
	db_wthno_na = db_wthno_na.replace('na', 0)

	#Replace 'na' values to avg of current column
	col_size = db.shape[1]
	for column in db:
	    naCount = len(db[db[column] == 'na'])
	    db_wthno_na[column] = pd.to_numeric(db_wthno_na[column])
	    db_sum = db_wthno_na[column].sum()
	    avg = db_sum/(col_size - naCount)
	    db[column] = db[column].replace('na', avg)

	if 'class' in db.columns:
		X = db.drop("class", axis = 1).copy()
		y = db['class'].copy()
	else:
		X = db.copy()
		y = []

	return X, y

def preprocess_XData(XData):
	XData = XData.drop("objid", axis = 1)
	XData = XData.drop(columns=['ra', 'dec','rowc','colc', 'rowv', 'colv', 'u_1', 'g_1', 'r_1', 'z_1', 'r_2', 'i_1', 'u_2', 'z_2', 'g_2', 'i_2'])
	return XData

def main():
	if(len(sys.argv) != 5):
		print("Arguments are not enough, command should be \'python3 main.py [path_to_train] [path_to_unlabeled_data] [path_to_test] [path_to_predictions]\'")
		sys.exit(-1)
	
	train_db    = pd.read_csv(sys.argv[1])
	unlabled_db = pd.read_csv(sys.argv[2])
	test_db     = pd.read_csv(sys.argv[3])

	path_to_predictions = sys.argv[4]

	X_train, y_train       = split_data(train_db)
	X_unlabled, y_unlabled = split_data(unlabled_db)
	X_test, y_test         = split_data(test_db)

	X_train    = preprocess_XData(X_train)
	X_unlabled = preprocess_XData(X_unlabled)
	X_test     = preprocess_XData(X_test)


	model = RandomForestClassifier(max_depth=30, n_estimators=100, max_features='auto')
	model.fit(X_train, y_train)

	print("Test accuracy for default forest:", model.score(X_test, y_test))
	print("F1 macro score:", f1_score(y_test, model.predict(X_test), average='macro'))

	y_unlabled = model.predict(X_unlabled)

	pred_df = pd.DataFrame({'objid':unlabled_db['objid'], 'prediction':y_unlabled})
	pred_df.to_csv(path_to_predictions, index=False)



if __name__== "__main__":
	main()
