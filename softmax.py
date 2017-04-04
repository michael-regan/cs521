import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import numpy as np
import time


# load dataset
glove_path='/path/to/lemmas_glove.csv'

glove_no_punct_path='/path/to/lemmas_glove_no_punct.csv'

glove_psst_path='/path/to/lemmas_glove_psst.csv'

psst_nsst_path='/path/to/pssts+nssts_glove.csv'

seed = 7
np.random.seed(seed)

dataframe = pandas.read_csv(psst_nsst_path, header=None)
dataset = dataframe.values
X = dataset[:,0:500].astype(float)
Y = dataset[:,500]

prec_list=[]
rec_list=[]
f1_list=[]

print("Using PSSTs+NSSTs")
print('\n')

for iter in range(1):
    
    for i in [3000]:
    
        for hidden_layers in [500]:
            
            tic = time.clock()
                
            A = X[:i]
            B = Y[:i]

            num_output=len(np.unique(B))

            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(B)
            encoded_Y = encoder.transform(B)
            # convert integers to one hot encoded
            dummy_y = np_utils.to_categorical(encoded_Y)


            def baseline_model():
            	# Create model
            	model = Sequential()
            	model.add(Dense(hidden_layers, input_dim=500, init='normal', activation='relu'))
            	model.add(Dense(num_output, init='normal', activation='sigmoid'))
            	# Compile model
            	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            	return model
    
            estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)

            X_train, X_test, Y_train, Y_test = train_test_split(A, dummy_y, test_size=0.05, random_state=seed)
            estimator.fit(X_train, Y_train)
            predictions = estimator.predict(X_test)

            #print(predictions)
            print('\n')
            X_pred=encoder.inverse_transform(predictions)

            max_indices=Y_test.argmax(axis=1)
            #print(max_indices)
    
            p_weighted=precision_score(max_indices, predictions, average='weighted')
            r_weighted=recall_score(max_indices, predictions, average='weighted')
            f1_weighted=f1_score(max_indices, predictions, average='weighted')
            
            prec_list.append(p_weighted)
            rec_list.append(r_weighted)
            f1_list.append(f1_weighted)

            print("Precision weighted: ", p_weighted)
    
            print("Recall weighted: ", r_weighted) 
    
            print("F1 weighted: ", f1_weighted) 

            #kfold = KFold(n=len(A), n_folds=10, shuffle=True, random_state=seed)

            #results = cross_val_score(estimator, A, dummy_y, cv=kfold)

            toc = time.clock()
            print("Size: ", i, "Hidden layers: ", hidden_layers, "Time: ", toc-tic, "secs")
            #print("Size data set:" , i, "Time: ", toc-tic, "secs")
            #print("Baseline (10-fold cv): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

print('\n')

print("Average precision: ", np.mean(prec_list))
print("Average recall: ", np.mean(rec_list))
print("Average f1: ", np.mean(f1_list))

print('\n')

print("precision list", prec_list)
print("recall list", rec_list)
print("f1 list", f1_list)

    








