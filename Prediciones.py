#Librerías
import tensorflow as tf #Sirve para crear una IA
import numpy as np #Sirve para operaciones matemáticas
import cv2 #Librería de OpenCV
from keras.preprocessing.image import img_to_array 


#---------DIRECCIONES DE LOS MODELOS------------
#ModeloDenso = r'C:\Users\moral\Desktop\IA\Programs\REDES NEURONALES CNN\ClasificadorDenso.h5'
#ModeloCNN = r'C:\Users\moral\Desktop\IA\Programs\REDES NEURONALES CNN\ClasificadorCNN.h5'
ModeloCNN2 = r'C:\Users\moral\Desktop\IA\Programs\REDES NEURONALES CNN\ClasificadorCNN2.h5'


#--------------LEEMOS LAS REDES NEURONALES--------------
#Denso
#Denso = tf.keras.models.load_model(ModeloDenso)
#pesosDenso = Denso.get_weights()
#Denso.set_weights(pesosDenso)
#CNN
#CNN = tf.keras.models.load_model(ModeloCNN)
#pesosCNN = CNN.get_weights()
#CNN.set_weights(pesosCNN)
#CNN2
CNN2 = tf.keras.models.load_model(ModeloCNN2)
pesosCNN2 = CNN2.get_weights()
CNN2.set_weights(pesosCNN2)


#---------------REALIZAMOS LA VIDEOCAPTURA------------
cap = cv2.VideoCapture(0)

# Aumentar FPS para mejorar fluidez
cap.set(cv2.CAP_PROP_FPS, 30)

#Empieza nuestro while true
while True:

    #Lectura de la videocaptura
    ret, frame = cap.read()

    #Lo pasamos a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Redimencionamos la imágen
    gray = cv2.resize(gray, (200, 200), interpolation = cv2.INTER_CUBIC)

    #Normalizamos la imágen
    gray = np.array(gray).astype(float) / 255

    #Convertimos la imágen en Matríz
    img = img_to_array(gray)
    img = np.expand_dims(img, axis=0)

    #Realizamos la Predicción
    prediccion = CNN2.predict(img)[0][0]
    print(prediccion)

    #Realizamos la Clasificación
    if prediccion <=0.5:
        cv2.putText(frame, "Gato", (200,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Perro", (200,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    
    #Mostramos los fotogramas
    cv2.imshow("CNN", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
cap.release()