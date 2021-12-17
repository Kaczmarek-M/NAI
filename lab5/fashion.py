'''
Adrian Matyszczak s19850
Michał Kaczmarek s18464

Moda MNIST ma służyć jako zamiennik dla klasycznej MNIST zbiór danych, często wykorzystywane
jako „Hello, World” programów uczenia maszynowego dla wizji komputerowej. Zbiór danych MNIST
zawiera obrazy odręcznych cyfr (0, 1, 2 itd.) w formacie identycznym z używanymi tutaj
artykułami odzieżowymi.

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Zbiór Fashon MINST z biblioteki Tensoflow
from tensorflow.python.ops.confusion_matrix import confusion_matrix

fashion_mnist = tf.keras.datasets.fashion_mnist
# przypisanie danych do zbiorów
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Każdy obraz jest mapowany na pojedynczą etykietę. Od nazwy klasy nie są dołączone do zestawu danych,
# zapisz je tutaj, aby później wykorzystać podczas drukowania obrazów
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                'Sneaker', 'Bag', 'Ankle boot']


train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

'''
Dane muszą zostać wstępnie przetworzone przed uczeniem sieci. Wartości pikseli mieszczą się w zakresie
między 0 a 255.

'''

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

'''
Przeskalowywanie wartości do zakresu od 0 do 1 i należy je podzielić przez 255 
przed wprowadzeniem ich do modelu sieci neuronowej
'''
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
Wyświetlanie 25 pierwszych zdjęć aby sprawdzić czy dane są w odpowiednim
formacie i czy sieć jest gotowa do nauczania
'''

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

'''
Budowanie modelu. Warstwy sieci neuronowej wyodrębniają reprezentację wprodzonych 
do nich danych.
'''
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
'''
Kompilacja modelu
'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

'''
Trenowanie modelu, wprowadzanie danych szkoleniowych, uczenie modelu kojarzenia
obrazów i etykiet w 10 epocs.
'''
model.fit(train_images, train_labels, epochs=10)
'''
Sprawdzamy, czy wytrenowany model zgadza się z testowym
'''
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

'''
Po przeszkoleniu modelu można go używać do prognozowania niektórych obrazów.
'''

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

predictions[0] # Model przewidział etykietę dla każdego obrazu w zestawie testowym

np.argmax(predictions[0]) # Przewidywana tablica to 10 liczb reprezętujących właściwość dobrania odpowiedniej etykiety do obrazu

test_labels[0]

'''
Tworzenie wykresu
'''
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


'''
Sprawdzanie prognoz po przeszkoleniu modelu. Prawidłowe etykiety prognozy są niebieskie,
a nieprawidłowe etykiety prognoz są czerwone
'''
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

'''
Po przeszkoleniu modelu można go używać do prognozowania niektórych obrazów.
Prawidłowe etykiety prognozy są niebieskie, a nieprawidłowe etykiety prognoz są czerwone.
'''

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
      plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
      plot_image(i, predictions[i], test_labels, test_images)
      plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
      plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

'''
Prognoza dotycząca pojedynczego obrazu
'''

img = test_images[15] # Pobieranie danych z zbioru zadań (testowego).

print(img.shape)

img = (np.expand_dims(img, 0)) # Dodawanie obrazu

print(img.shape)

predictions_single = probability_model.predict(img) # Typowanie poprawnej etykiety obrazu

'''
Typowanie poprawnej etykiety dla obrazu.
'''
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

img = test_images[1]

print(img.shape)

img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])

y_pred = model.predict(test_images)
y_p=np.argmax(y_pred, axis=1)
print(confusion_matrix(test_labels,y_p))
'''
Tablica pomyłek
'''
mat = confusion_matrix(test_labels, y_p)
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=mat)
disp = disp.plot()
plt.show()




