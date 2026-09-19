# TFG-RL-Spacecraft-Rendezvous

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Grado sobre la aplicación de técnicas de Reinforcement Learning (RL) al guiado autónomo de una nave espacial durante una maniobra de rendezvous. El proyecto utiliza Stable-Baselines3 (SB3) y Gymnasium para el desarrollo, entrenamiento y evaluación de agentes de aprendizaje por refuerzo.

La carpeta CartPole contiene los archivos correspondientes a las pruebas iniciales realizadas sobre este entorno, mientras que Rendezvous contiene la implementación del entorno de rendezvous, los scripts de entrenamiento y evaluación y los experimentos de Monte Carlo.

## Instalación

Para ejecutar el proyecto se recomienda utilizar un entorno virtual de Python. El proyecto ha sido desarrollado utilizando Python 3.9.13. Las versiones de las principales dependencias utilizadas se encuentran especificadas en el archivo requirements.txt.

### 1. Crear el entorno virtual

Desde la carpeta raíz del repositorio, se crea un entorno virtual mediante:

python -m venv venv
### 2. Activar el entorno virtual

En Windows:

venv\Scripts\activate

En Linux/macOS:

source venv/bin/activate

Una vez activado, el nombre del entorno virtual aparecerá al principio de la línea de comandos.

### 3. Instalar las dependencias

Con el entorno virtual activado, se instalan las dependencias especificadas en requirements.txt mediante:

pip install -r requirements.txt

Este comando instalará las versiones de las librerías necesarias para ejecutar el proyecto.

### 4. Comprobar la versión de Python

Para comprobar que se está utilizando la versión de Python adecuada:

python --version

La versión obtenida debería corresponder con la versión utilizada durante el desarrollo del proyecto:

Python 3.9.13
### 5. Comprobar la instalación de Stable-Baselines3

Para comprobar que Stable-Baselines3 se ha instalado correctamente:

python -c "import stable_baselines3; print(stable_baselines3.__version__)"

También se puede comprobar la instalación de Gymnasium mediante:

python -c "import gymnasium; print(gymnasium.__version__)"

## Ejecución

Una vez creado y activado el entorno virtual e instaladas las dependencias, los diferentes scripts pueden ejecutarse desde la carpeta correspondiente del repositorio.

Los scripts de entrenamiento permiten generar nuevos agentes de aprendizaje por refuerzo, mientras que los scripts de evaluación permiten analizar su comportamiento en el entorno de rendezvous.

Para la fase de postprocesado, es importante volver a copiar el archivo de settingsRL.txt en la carpeta que se obtiene del modelo entrenado (Xenv_Ynb_ZZstep_1) para poder ejecutar los códigos main_rendezvous_MC.py, plot_training_curves.py y main_rendezvous_load.py. En estos códigos también es importante poner la ruta del entrenamiento correctamente.

## Dependencias

Las principales herramientas y librerías utilizadas en el proyecto son:

Python
Stable-Baselines3
Gymnasium
NumPy
Matplotlib
SciPy

Las versiones concretas de las dependencias se encuentran definidas en el archivo requirements.txt.

## Reproducibilidad

El archivo requirements.txt permite especificar las versiones de las dependencias utilizadas durante el desarrollo del proyecto. Junto con la versión de Python indicada anteriormente, esto facilita la creación de un entorno de ejecución equivalente al empleado durante los experimentos.

Para reproducir el entorno de desarrollo, basta con crear un entorno virtual utilizando la versión de Python indicada e instalar las dependencias mediante:

pip install -r requirements.txt

Los resultados presentados en el Trabajo de Fin de Grado se obtienen mediante la ejecución de los scripts de entrenamiento y evaluación incluidos en este repositorio.
