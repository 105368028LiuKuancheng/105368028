import pandas as pd
import numpy as np



# Read dataset into X and Y
df = pd.read_csv('/home/eee/HW1/train-v3.csv')
dataset = df.values
X_train = dataset[:, 2:22]
Y_train = dataset[:, 1]

df2 = pd.read_csv('/home/eee/HW1/test-v3.csv')
dataset2=df2.values
X_test = dataset2[:,1:21]

df3 = pd.read_csv('/home/eee/HW1/valid-v3.csv')
dataset3=df3.values
X_valid = dataset3[:, 2:22]
Y_valid = dataset3[:, 1]



#print "X: ", X
#print "Y: ", Y


# Define the neural network
from keras.models import Sequential
from keras.layers import Dense

def normalize(train,valid,test):
# tmp = np.concatenate((train,valid),axis=0)
    tmp = train
    mean,std = tmp.mean(axis=0), tmp.std(axis=0)
    print("tmp.shape= ", tmp.shape)
    print("mean.shape= ", mean.shape)
    print("std.shape= ", std.shape)
    print("mean= ", mean)
    print("std= ", std)
    train = (train - mean) / std
    valid = (valid - mean) / std
    test = (test - mean) / std
    return train, valid ,test


X_train, X_valid, X_test = normalize(X_train, X_valid, X_test)

def build_nn():
    model = Sequential()
    model.add(Dense(32, input_dim=20, init='normal', activation='relu'))
    model.add(Dense(70, input_dim=32, init='normal', activation='relu'))
    model.add(Dense(128, input_dim=70, init='normal', activation='relu'))
    model.add(Dense(128, input_dim=128, init='normal', activation='relu'))
    model.add(Dense(70, input_dim=128, init='normal', activation='relu'))
    model.add(Dense(32, input_dim=70, init='normal', activation='relu'))
    model.add(Dense(20, input_dim=32, init='normal', activation='relu'))
    #model.add(Dense(32, input_dim=60, init='normal', activation='relu'))
    # No activation needed in output layer (because regression)
    model.add(Dense(1, init='normal'))

    # Compile Model
    model.compile(loss='MAE', optimizer='adam')
    model.fit(X_train, Y_train, epochs=200, batch_size=32,validation_data = (X_valid,Y_valid))
    Y_predict = model.predict(X_test)
    np.savetxt('test.csv',Y_predict,delimiter = ',')


    return model


# Evaluate model (kFold cross validation)
from keras.wrappers.scikit_learn import KerasRegressor

# sklearn imports:
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Before feeding the i/p into neural-network, standardise the dataset because all input variables vary in their scales
estimators = []
estimators.append(('standardise', StandardScaler()))
estimators.append(('multiLayerPerceptron', KerasRegressor(build_fn=build_nn, nb_epoch=100, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n=len(X_train), n_folds=10)
results = cross_val_score(pipeline, X_train, Y_train, cv=kfold)


print ("total: ", results.sum())
print ("Mean: ", results.mean())
print ("StdDev: ", results.std())
