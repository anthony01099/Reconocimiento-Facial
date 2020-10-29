## Face recognition system

 Este repositorio implementa una clase que permite crear vectores de dimensión 128 a partir de imagenes de personas. El algoritmo recorta el rostro de la imagen (el rostro más grande en la imagen), y lo procesa con una red neuronal que tiene como salida dicho vector. El vector resultante comprime todas las características propias del rostro, de manera que la identificación facial se puede resolver como un simple problema de clasificación.

 ## Testing

 Para probar el algoritmo solo necesita tener instalado python junto con las librerías en requirements.txt y ejecutar el archivo test.py.

    python test.py

Este programa empezara a tomar fotos de usted y le mostrará la distancia euclidea entre los vectores formados a partes de imagenes seguidas. Si otra persona se coloca en frente de la cámara, usted debería notar como esa distancia euclidea aumenta.

## Implementación del reconocimiento facial

Aún falta implementar el clasificador, sin embargo no es una tarea muy compleja. Siga los siguientes pasos:
- Cree una base de datos con identificadores para usuarios, y almacene diversos vectores de rostro de cada usuario debidamente identificados.
- Para identificar una imagen nueva, procese la misma con el algoritmo dado acá, y use la base de datos para identificar al usuario. Eso lo puede hacer haciendo uso del algoritmo K Nearest Neighbors (KNN). Uso la librería scikit learn para la implementación https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html. También puede emplear una máquina de vector suporte (SVM), sin embargo puede ser muy exigente computacionalmente.
- El clasificador (KNN o SVM) debe entrenarse con todos los vectores en la base de datos al iniciarse el programa o al añadir una nueva imagen. El clasificador entrenado debe usarse para identificar el usuario.
- Debe desarrollar la interfaz con el usuario para:
    - Cargar una imagen de usuario en la base de datos (no se almacena la imagen, sino el vector).
    - Eliminar una imagen.
    - Identificar una imagen.
- Dependiendo de la cantidad de datos a manejar, se recomienda una implementación usando Django como framework web y PostgresSQL como gestor de base de datos. O puede integrarse al proyecto ApiCabina que hace uso de estas tecnologías.
- Una implementación viable sería enviar mediante un POST la imagen encriptada en base 64, y el servidor aplica el algoritmo y retorna un json con los datos del usuario.
