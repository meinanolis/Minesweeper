import numpy as np 
from sklearn import svm
import pandas as pd 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pickle

#df_train=pd.read_csv('train_data (xpos,ypos,...).csv')
df_train=pd.read_csv('train_data.csv')
train=df_train.values
X=train[:,:-1]/10
X[:,:2]=train[:,:2]/30
#y=np.vstack((train[:,-1],1-train[:,-1])).T
y=train[:,-1]
print(X.shape,y.shape)

test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)


filename = 'model2.sav'
if 1:
	#model = LogisticRegression()
	#model = svm.LinearSVC()
	model = clf = MLPClassifier(alpha=1e-4, hidden_layer_sizes=(10000, 1000, 1000, 1000, 300))
	model.fit(X_train, Y_train)
	pickle.dump(model, open(filename, 'wb'))
else:
	model = pickle.load(open(filename, 'rb'))


result = model.score(X_test, Y_test)
print(result)



yt=Y_test[10:40]
yp=model.predict(X_test[10:40])

print(np.vstack((yt,yp)))