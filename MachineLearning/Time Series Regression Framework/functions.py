import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error

from classes import ToSupervised, ToSupervisedDiff


def rmsle(ytrue, ypred):
	return np.sqrt(mean_squared_log_error(ytrue, ypred))


def getDataFramePipeline(i):
	steps = [(str(i) + '_step', ToSupervised('Sales', 'Product_Code', i))]
	for j in range(1, i + 1):
		if i == j:
			
			pp = (str(j) + '_step_diff',
			      ToSupervisedDiff(str(i) + '_Week_Ago_Sales', 'Product_Code', 1, dropna=True))
			steps.append(pp)
		else:
			
			pp = (str(j) + '_step_diff',
			      ToSupervisedDiff(str(i) + '_Week_Ago_Sales', 'Product_Code', 1))
			steps.append(pp)
	
	return steps


from tqdm import tqdm


def stepsTune(X, model, num_steps, init=1):
	scores = []
	for i in tqdm(range(init, num_steps + 1)):
		steps = []
		steps.extend(getDataFramePipeline(i))
		steps.append(('predic_1', model))
		super_ = Pipeline(steps).fit(X)
		score_ = np.mean(super_.score(X))
		scores.append((i, score_))
	
	return scores

