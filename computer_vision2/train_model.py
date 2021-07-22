import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import h5py
import matplotlib.pyplot as plt
import scikitplot
db = h5py.File('/home/quan/PycharmProjects/DL_Python/computer_vision2/data/animals.hdf5','r')
i = int(db['labels'].shape[0]*0.75)

print("[INFO] tuning hyperparameters...")
params = {'C':[0.1 , 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=-1)

model.fit(db['features'][:i], db['labels'][:i])
print("[INFO] best hyperparameters:{}".format(model.best_params_))


preds = model.predict(db['features'][i:])
print(classification_report(db['labels'][i:], preds, target_names=db['label_names']))




y_pred = np.argmax(preds)
t_te = db['labels'][i:]


scikitplot.metrics.plot_confusion_matrix(t_te, preds, figsize=(8,8))
plt.show()
f = open('animal.cpickle', 'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close()
db.close()