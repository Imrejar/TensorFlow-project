import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


import matplotlib.pyplot as plt

# Définir les noms des classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Afficher les premières images de l'ensemble d'entraînement avec leurs étiquettes
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()



# Affichage des dimensions des données
print("Dimensions des données d'entraînement :", train_images.shape)
print("Dimensions des étiquettes d'entraînement :", train_labels.shape)

# Affichage d'une image d'exemple
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()



train_images = train_images / 255.0
test_images = test_images / 255.0



model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))



test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Précision sur les données de test :", test_acc)

model.save("models/your_model.h5")

predictions = model.predict(test_images)



from PIL import Image
import numpy as np

# Charger l'image PNG
image = Image.open("canvas.png")

# Redimensionner l'image à la taille requise (par exemple, 28x28 pour Fashion MNIST)
image = image.resize((28, 28))

# Convertir l'image en niveau de gris
image = image.convert("L")

# Convertir l'image en tableau numpy
new_image = np.array(image)

# Inverser les couleurs si nécessaire (noir sur blanc -> blanc sur noir)
new_image = 255 - new_image

# Normaliser les valeurs des pixels entre 0 et 1
new_image = new_image / 255.0

# Redimensionner l'image pour qu'elle ait une dimension supplémentaire correspondant au lot de données
new_image = np.expand_dims(new_image, axis=0)





# Faire la prédiction sur la nouvelle image
predictions = model.predict(new_image)

# Afficher la prédiction
predicted_class = np.argmax(predictions)
print('Classe prédite :', class_names[predicted_class])






from PIL import Image
import numpy as np

# Charger l'image PNG
image = Image.open("Pull.jpg")

# Redimensionner l'image à la taille requise (par exemple, 28x28 pour Fashion MNIST)
image = image.resize((28, 28))

# Convertir l'image en niveau de gris
image = image.convert("L")

# Convertir l'image en tableau numpy
pull = np.array(image)

# Inverser les couleurs si nécessaire (noir sur blanc -> blanc sur noir)
pull = 255 - pull

# Normaliser les valeurs des pixels entre 0 et 1
pull = pull / 255.0

# Redimensionner l'image pour qu'elle ait une dimension supplémentaire correspondant au lot de données
pull = np.expand_dims(pull, axis=0)
















# Faire la prédiction sur la nouvelle image
predictions = model.predict(pull)

# Afficher la prédiction
predicted_class = np.argmax(predictions)
print('Classe prédite :', class_names[predicted_class])











from PIL import Image
import numpy as np

# Charger l'image PNG
image = Image.open("trouser.png")

# Redimensionner l'image à la taille requise (par exemple, 28x28 pour Fashion MNIST)
image = image.resize((28, 28))

# Convertir l'image en niveau de gris
image = image.convert("L")

# Convertir l'image en tableau numpy
trouser = np.array(image)

# Inverser les couleurs si nécessaire (noir sur blanc -> blanc sur noir)
trouser = 255 - trouser

# Normaliser les valeurs des pixels entre 0 et 1
trouser = trouser / 255.0

# Redimensionner l'image pour qu'elle ait une dimension supplémentaire correspondant au lot de données
trouser = np.expand_dims(trouser, axis=0)

# Faire la prédiction sur la nouvelle image
predictions = model.predict(trouser)

# Afficher la prédiction
predicted_class = np.argmax(predictions)
print('Classe prédite :', class_names[predicted_class])
