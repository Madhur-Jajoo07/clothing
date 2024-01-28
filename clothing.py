#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K



# %%
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# %%
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)


#%%
plt.figure(figsize=(20, 10))
for i in range(16):
  plt.subplot(4, 4, i + 1)
  plt.axis('off')
  plt.imshow(X_train[i], cmap='gray')
  plt.title(y_train[i])

plt.show()

#%%
rows = X_train[0].shape[0]
cols = X_train[1].shape[0]

X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
X_test = X_test.reshape(X_test.shape[0], rows, cols, 1)

inp_shape = (rows, cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_cls = y_test.shape[1]
num_pix = X_train.shape[1] * X_train.shape[2]


#%%
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=inp_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_cls, activation='softmax'))


#%%

model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])
#%%
model.summary()
#%%
batch_size = 128
epochs = 100 

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
#%%
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title("Accuracy Vs. Epcohs Graph", fontsize=20)
#%%
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Accuracy')
plt.plot(history.history['val_loss'], label='Val Accuracy')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title("Loss Vs. Epcohs Graph", fontsize=20)
#%%
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
#%%
test = []
for x in y_test:
  if 1 in x:
    x = list(x)
    test.append(x.index(1))
y_test = np.array(test)
print("Accuracy Score: ", accuracy_score(y_test, pred))
