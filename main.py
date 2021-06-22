import numpy as np
from keras.constraints import maxnorm
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = loadtxt('breast-cancer-wisconsin.csv', delimiter =',');
#print(dataset.shape)

X = dataset[:,1:10]
y = dataset[:,10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

X = (X-np.min(X))/(np.max(X)-np.min(X))


#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#print(X.shape, y.shape)
print(X)

print(y)

#define Model
model = Sequential()
model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(33, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile Model - KERAS MODEL
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=100)
print(history.history.keys())

print(model.predict(X_test))
#evaloate Model
pred = (model.predict(X_test) > 0.5).astype("int32")
print(y_test)
print(pred)


#Gráfico accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Gráfico loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




cm = confusion_matrix(y_test, pred)
print(cm)

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel();

tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
acc = (tp + tn) / (tp + tn + fn + fp)

print('TPR: ', tpr);
print('TNR: ', tnr);
print('ACCURACY', acc);

#AUC
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
auc = metrics.auc(fpr, tpr)


#Gráfico curva ROC
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.plot(fpr, tpr, color='b', label=fr'ROC (AUC = {auc:0.2F})', lw=2, alpha=.8)
plt.suptitle('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.show()




