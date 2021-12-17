'''
Adrian Matyszczak s19850
Michał Kaczmarek s18464

Opis Problemu:
Nauczenie sieci neuronowej klasyfikacji danych.
'''
import self as self
import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# pobieranie i przygotowywanie zbioru CIFAR10

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#N ormalizujemy dane dzieląc je przez 255
train_images, test_images = train_images / 255.0, test_images / 255.0

# weryfikacja danych, nazwy klas

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

'''
Wyświetlanie przykładowych zdjęć z bazy danych
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # Etykiety CIFAR są tablicami, dlatego potrzbujemy dodatkowego indexu
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

'''
Definiujemy model, ten model jest odpowiedni dla zwykłego stosu warstw, gdzie każda warstwa
ma jeden tesor wejściowy i jeden wyjściowy.
'''

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Pokażmy dotychczasową architekturę modelu

model.summary()

# Dodaj gęste warstwy na wierzchu

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# pełna architektura modelu

model.summary()

# Kompilacja i trenowanie modelu

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# ocena modelu

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


print(test_acc)

