'''
Adrian Matyszczak s19850
Michał Kaczmarek s18464

Nasze rozwiązanie klasyfikuje obiekty ze zdjęcia do jednej z tysiąca możliwych kategorii.
Wykorzystujmy model ResNet50, nauczony na zbiorze danych imagenet, w celu szybkiej klasyfikacji nowych obrazów.

'''

from IPython.display import Image
from tensorboard.notebook import display
'''
Przykładowe zdjęcie do kwalifikacji
'''
Image(filename='pies.jpg')
Image(filename='warplane.png')
Image(filename='zamek.jpg')
Image(filename='hotdog.jpg')


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

img_path = 'pies.jpg'
#img_path = 'zamek.jpg'
#img_path = 'hotdog.jpg'

'''
Skalowanie zdjęca do 224x224 pix.
Załadowanie obrazu.  
'''
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)   # Normalizacja danych obrazu przed wprowadzeniem ich jako danych wejściowych

model = ResNet50(weights='imagenet')  # ładowanie modelu

preds = model.predict(x)

# dekoduj wynik na listę (klasa, opis, prawdopodobieństwo), wyświetlanie 3 najlepszych wyników
print('Predicted:', decode_predictions(preds, top=3)[0])



