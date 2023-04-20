from nltk import word_tokenize
from nltk import WordNetLemmatizer
import numpy as np
import json
import pickle
import random

lemitizer = WordNetLemmatizer()


word = []
classes = []
documents = []
ignore = ["!","?","."]

with open("intents.json") as file:
    file = json.load(file)

for intent in file["intents"]:
    for pattern in intent["patterns"]:
        w = word_tokenize(pattern)
        word.extend(w)
        documents.append((w,intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

word = [lemitizer.lemmatize(w.lower()) for w in word if w not in ignore]

word = sorted(list(set(word)))

classes = sorted(list(set(classes)))

pickle.dump(word,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_word = doc[0]

    pattern_word = [lemitizer.lemmatize(words.lower()) for words in pattern_word]

    for w in word:
        bag.append(1)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)


train_x = list(training[:,0])
train_y = list(training[:,1])

import tensorflow

model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.layers.Dense(units=128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(tensorflow.keras.layers.Dropout(0.5))
model.add(tensorflow.keras.layers.Dense(units=64, activation="relu"))
model.add(tensorflow.keras.layers.Dropout(0.5))
model.add(tensorflow.keras.layers.Dense(len(train_y[0]), activation="softmax"))

model.compile(tensorflow.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(np.array(train_x), np.array(train_y), batch_size=10, epochs=2000, verbose=1)
model.save("model.h5")