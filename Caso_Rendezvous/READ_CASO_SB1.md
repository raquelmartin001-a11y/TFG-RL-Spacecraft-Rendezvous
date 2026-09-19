# Guía de instalación y ejecución del proyecto DL‑rendezvous‑guidance

Enlace: https://github.com/LorenzoFederici/DL-rendezvous-guidance

Este documento explica cómo instalar y ejecutar el proyecto en cualquier ordenador, incluso si no dispone de GPU moderna o si necesita instalar TensorFlow 1.15 mediante un archivo .whl. También se detalla la instalación de MPI, requisito fundamental para ejecutar los scripts distribuidos del repositorio.

## Requisitos previos 
  -  Python 3.7 (TensorFlow 1.15 no funciona en versiones superiores)
  -  pip
  -  virtualenv o venv
  -  OpenMPI
  -  Archivo .whl de TensorFlow 1.15 (CPU o GPU según tu caso)

## Crear entorno virtual

  python3.7 -m venv dl_env
  source dl_env/bin/activate

## Instalar MPI 

