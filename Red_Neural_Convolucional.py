#Importar Librerías

import os #Sirve para llamar el Dataset
import tensorflow as tf #Sirve para crear una IA
import numpy as np #Sirve para operaciones matemáticas
import matplotlib.pyplot as plt #Sirve para realizar gráficas
import cv2 #Librería de OpenCV
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau #Librería para observar el funcionamiento de la Red
from tensorflow.keras.preprocessing.image import ImageDataGenerator #Sirve para modificar las imágenes, y hacer un entrenamiento mas "Realista"
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay


#Almacenamos la dirección del dataset/información a ocupar
entrenamiento = r'C:\Users\moral\Desktop\IA\Programs\REDES NEURONALES CNN\Dataset\Entrenamiento'
validacion = r'C:\Users\moral\Desktop\IA\Programs\REDES NEURONALES CNN\Dataset\Validacion'


#llamamos la información que existe en cada carpeta del DataSet
listaTrain = os.listdir(entrenamiento) 
listaTest =os.listdir(validacion)


#Se establece algunos parámetros
ancho, alto = 200, 200


#Lista de entrenamiento
etiquetas = []
imagenes = [] #Neuronas simples
datos_train = []
conv = 0 #Neuronas Complejas

#Lista de validación
etiquetas2 = []
imagenes2 = [] #Neuronas simples
datos_test = []
conv2 = 0 #Neuronas complejas


#Extraer en una lista las imágenes y agregar las etiquetas
#Entrenamiento
for nameDir in listaTrain:
    nombre = entrenamiento + '/' + nameDir #Leemos las imágenes

    for fileName in os.listdir(nombre): #Asignamos las etiquetas a cada imagen
        etiquetas.append(conv) #Valor de la etiqueta (aqui le asignamos 0 a la primer carpeta y 1 a la segunda carpeta)
        img = cv2.imread(nombre + '/' + fileName, 0) #Leemos la imágen y la cargamos a escalas de grises
        if img is None:  # Si la imagen no se carga correctamente
            print(f"Error al cargar: {nombre + '/' + fileName}")  
            continue  # Salta esta imagen y sigue con la siguiente

        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC) #Redimencionamos las imágenes
        img = np.reshape(img, (ancho, alto, 1)) #Cambiamos la forma de la imágenpara agregarlo a un canal
        datos_train.append([img, conv])
        imagenes.append(img) #Añadimos las imágenes en EDG

    conv = conv + 1


#Validación
for nameDir2 in listaTest:
    nombre2 = validacion + '/' + nameDir2 #Leemos las imágenes

    for fileName2 in os.listdir(nombre2): #Asignamos las etiquetas a cada imagen
        etiquetas2.append(conv2) #Valor de la etiqueta (aqui le asignamos 0 a la primer carpeta y 1 a la segunda carpeta)
        img2 = cv2.imread(nombre2 + '/' + fileName2, 0) #Leemos la imágen
        img2 = cv2.resize(img2, (ancho, alto), interpolation=cv2.INTER_CUBIC) #Redimencionamos las imágenes
        img2 = np.reshape(img2, (ancho, alto, 1)) #Cambiamos la forma de la imágen para agregarlo a un canal
        datos_test.append([img2, conv2])
        imagenes2.append(img2) #Añadimos las imágenes en EDG

    conv2 = conv2 + 1


#Vamos a normalizar las imágenes (0 a 1)
#El objetivo es que en vez de tener una escala de grises del 0-255, poder tener una del 0-1, y los valores que se encuentran en medio, 
#pasarlos a valores flotantes.
imagenes = np.array(imagenes).astype(float) / 255
print(imagenes.shape)
imagenes2 = np.array(imagenes2).astype(float) / 255
print(imagenes2.shape)


#Pasamos las lista a Matrices
etiquetas = np.array(etiquetas).astype(np.float32)
etiquetas2 = np.array(etiquetas2).astype(np.float32)


#Vamos asignar un entrenamiento "Realista", para que el programa aprenda no sólo viendo las imágnes, sino viendo el cambio que existe al
# momento de rotarlas, hacerles zoom, moverlas, etc., con el fin de que en la hora de la validación el resultado sea satisfactorio.
imgTrainGen = ImageDataGenerator(
    rotation_range = 20,          #Rotación aleatoria
    width_shift_range = 0.15,      #Mover la imágen a los lados
    height_shift_range = 0.1,     #Mover la imágen de arriba y abajo
    shear_range = 0.2,            #Inclinar la imágen
    zoom_range = 0.2,      #Hacer zoom a la imágen
    vertical_flip = True,         #Giros aleatorios verticales
    horizontal_flip = True,        #Giros aleatorios horizontales
    brightness_range=[0.8, 1.2],  #Cambia el brillo
    fill_mode='nearest'           #Rellena áreas tras transformaciones
)

#Ahora lo aplicamos a las imágenes
imgTrainGen.fit(imagenes)
imgTrain = imgTrainGen.flow(imagenes, etiquetas, batch_size=16)


#---------------INICIAMOS CON LA ESTRUCTURA DE LA RED NEURONAL CONVOLUCIONAL-----------------
#Modelo con Capas Convolucionales y Drop out
#El objetivo de este modelo, es crear una red neuronal en la cual no tenga que aprender toda la información de golpe, es decir; lo que hará es
#ir prendiendo/apagando conexiones aleatoriamente para que vaya aprendiendo nuevos caminos/aprendizajes, y no se quede con el mismo camino de 
#conexión. En resumen, es lograr que vaya aprendiendo de todas las formas posibles sin necesidad de que el aprendizaje sea repetivo.
ModeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', input_shape = (200, 200, 1)), #Capa de entrada convolucional de 32 kernel
    tf.keras.layers.BatchNormalization(),                                                #Normalización para mejor entrenamiento
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2,2),                                                   #Capa de agrupación
    
    tf.keras.layers.Conv2D(128, (3,3), padding = 'same'),                              
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),                                             
    tf.keras.layers.MaxPooling2D(2,2),                                                  
    
    tf.keras.layers.Conv2D(256, (3,3), padding = 'same'),                             
    tf.keras.layers.BatchNormalization(),   
    tf.keras.layers.Activation('relu'),                                            
    tf.keras.layers.MaxPooling2D(2,2),        

    tf.keras.layers.Conv2D(512, (3,3), padding = 'same'),                             
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Activation('relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
                                                                                    

    #Capas Densa de Clasificación
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, kernel_regularizer=l2(0.001)), #Regularización l2(0.1 → 0.01) para evitar penalización excesiva   
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5), #El caambio sea de mitad en mitad (por eso se pone 0.5),
    tf.keras.layers.Dense(128), #Agregamos otra capa
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])


#---------------COPILAMOS LOS MODELOS Y AGREGAMOS EL OPTIMIZADOR Y LA FUNCIÓN DE PÉRDIDA-------------------
ModeloCNN2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss='binary_crossentropy', #Se usa porque tenemos 2 clases
                   metrics=['accuracy'])


#-------------OBSERVAMOS Y ENTRENAMOS LAS REDES-----------
#Para visualizarlo en CMD: tensorboard --logdir="C:\Users\moral\Desktop\IA\Programs\REDES NEURONALES CNN"

#-----------------------------------CALLBACKS-------------------------------------------
#Detendrá el entrenamiento si val_loss no mejora en 10 épocas.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#Entrenamos Modelo Convolucional y DropOut
BoardCNN2 = TensorBoard(log_dir=r'C:\Users\moral\Desktop\IA\Programs\REDES NEURONALES CNN')
history = ModeloCNN2.fit(imgTrain, batch_size = 16, validation_data = (imagenes2, etiquetas2),
               epochs = 100, callbacks = [BoardCNN2, early_stopping], steps_per_epoch = int(np.ceil(len(imagenes) / float(32))),
               validation_steps = int(np.ceil(len(imagenes2) / float(32))))

#Guardamos el modelo
ModeloCNN2.save('ClasificadorCNN2.h5')
ModeloCNN2.save_weights('pesosCNN2.weights.h5')
print("Terminamos Modelo CNN2")


#Crear una figura con 2 subgráficas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#Gráfica de pérdida (Loss)
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Evolución de la Pérdida')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('Loss')
axes[0].legend()

#Gráfica de precisión (Accuracy)
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Evolución de la Precisión')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

#Mostrar las gráficas
plt.show()