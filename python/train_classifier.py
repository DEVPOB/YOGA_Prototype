import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
with open('data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

data = data_dict['data']
labels = data_dict['labels']

max_sequence_length = max(len(seq) for seq in data)

for i in range(len(data)):
    if len(data[i]) < max_sequence_length:
        data[i] = np.pad(data[i], (0, max_sequence_length - len(data[i])))
    elif len(data[i]) > max_sequence_length:
        data[i] = data[i][:max_sequence_length]

data = np.array(data)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print("{}% Classified correct".format(score))

f = open('model.p', 'wb')

with open('model.p', 'wb') as file:
    pickle.dump(model, file)
f.close();