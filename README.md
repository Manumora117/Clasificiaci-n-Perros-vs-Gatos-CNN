# Clasificiacion-Perros-vs-Gatos-CNN
Red neuronal convolucional (CNN) para clasificación de imágenes de perros y gatos. El modelo utiliza un dataset de 4800 imágenes por clase (9600 totales) con división 50% entrenamiento / 50% pruebas.

## 📋 Contenido
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Dataset](#dataset)
- [Licencia](#licencia)

## 🧰 Requisitos
- Python 3.8+
- Dependencias principales:
  ```text
  tensorflow==2.12.0
  opencv-python==4.7.0
  numpy==1.24.3
  matplotlib==3.7.1

## ✅ Instlación del Proyecto
1. Clona el repositorio: git clone https://github.com/Manumora117/analisis-hospitalizaciones-pandemia-cdmx-edomex.git
2. Abre el archivo `Red_Neuronal_Convolucional.py` con python 3.8 o más.
3. Cambia la dirección en donde se encuentra 
4. Tener Instaladas las librerías de pandas, numpy, matplotlib.pyplot, seaborn y en caso de usarlo en Visual Studio Code, instalar ipykernel, para que lo pueda leer y correr.
5. RECOMENDACIÓN (Opcional): agregar un entorno virtual.

## ✅ Uso
1. Abre el archivo `Predicciones.py`.
2. Cambiar la direccion de donde se encuentra el archivo `ClasificadorCNN2.h5`, el cual se creo a la hora de correr el modelo.

## 🗂️ Dataset
El dataset lo pueden descargar desde la página de [Kaggle: Cat & Dogs]([https://datos.cdmx.gob.mx/](https://www.kaggle.com/datasets/d4rklucif3r/cat-and-dogs)) del usuario Arsh Anwar. Ahi puedem obtener el dataset que se utilizó para realizar este proyecto de clasificación de perros vs gatos.
*NOTA:* El tamaño del dataset que se utilizó para entrenar y validar (train/test), fue de 2400 imagenes para cada etiqueta, en resumen, se usó en total 4800 imagenes para cada etiqueta, haciendo unma repartición manual de 50/50. Aunque en el dataset que se encuentra en Kaggle, es de 5000 y esta dividido en 80/20: pueden utilizar el que viene por default, ya que los avlores que yo utilice fue sin tener conocimiento previo de la proporcionalidad de datos que debe tener (train/test).

## 🧠 Autor

Emmanuel Morales 
[LinkedIn](https://www.linkedin.com/in/tu-usuario/) | [GitHub](https://github.com/tuusuario)
