"""
Extrae y grafica las curvas de entrenamiento (learning curve, policy loss,
value loss y entropia) a partir de los logs de TensorBoard generados durante
el entrenamiento (requiere haber entrenado con tensorboard_log=... en
main_rendezvous_RL.py).

Este script NO depende directamente de stable-baselines ni de
stable-baselines3: lee los eventos de TensorBoard con
tensorboard.backend.event_processing, que es el mismo formato estandar en
ambas librerias. Por eso no necesita practicamente cambios de migracion,
salvo los NOMBRES de los tags por defecto, que SI cambian entre SB1 y SB3:

  Stable-Baselines (v1, TF)        Stable-Baselines3 (PyTorch)
  --------------------------       ----------------------------------
  episode_reward                   rollout/ep_rew_mean
  policy_gradient_loss             train/policy_gradient_loss
  value_function_loss              train/value_loss
  entropy                          train/entropy_loss

Uso:
    python plot_training_curves.py --folder sol_saved/8env_4nb_96Mstep_1/
    python plot_training_curves.py --folder sol_saved/8env_4nb_96Mstep_1/ \
        --keywords ep_rew_mean policy_gradient_loss value_loss entropy_loss

Si no estas seguro de los nombres exactos que escribio tu entrenamiento,
ejecuta el script sin --keywords: imprime todos los tags disponibles antes
de intentar graficar nada, para que puedas ajustar el filtro.
"""
import os
import math
import argparse
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="sol_saved/4env_4nb_10Mstep_3/",
    help='Carpeta de la solucion (out_folder / tensorboard_log de main_rendezvous_RL.py)')
parser.add_argument('--keywords', type=str, nargs='+',
    # Defaults actualizados a los nombres de tag que usa stable-baselines3.
    default=['ep_rew_mean', 'policy_gradient_loss', 'value_loss', 'entropy_loss'],
    help='Subcadenas (case-insensitive) para filtrar que tags de tensorboard graficar')
args = parser.parse_args()

# Buscar la carpeta con el fichero de eventos de tensorboard dentro de --folder
tb_dir = None
for root, dirs, files in os.walk(args.folder):
    if any(f.startswith('events.out.tfevents') for f in files):
        tb_dir = root
        break
if tb_dir is None:
    raise FileNotFoundError(
        "No se ha encontrado ningun fichero de eventos de tensorboard en " + args.folder +
        ". Comprueba que el entrenamiento se lanzo con tensorboard_log configurado "
        "(en SB3: model = PPO(..., tensorboard_log='ruta/')).")

print("Leyendo logs de tensorboard en: " + tb_dir)
ea = EventAccumulator(tb_dir)
ea.Reload()
tags = ea.Tags().get('scalars', [])

print("\nTags disponibles en tensorboard:")
for t in tags:
    print("  - " + t)

if not tags:
    raise RuntimeError("El fichero de eventos no contiene escalares. Revisa que el "
        "entrenamiento haya escrito logs correctamente.")

print("\nGenerando graficas...")
out_folder = args.folder

# Separar el/los keyword(s) de la curva de aprendizaje (reward) del resto
# (policy loss, value loss, entropia, etc.), que se consideran "perdidas"
reward_keywords = [k for k in args.keywords if 'rew' in k.lower()]
loss_keywords = [k for k in args.keywords if 'rew' not in k.lower()]

#--- Curva de aprendizaje (reward): un fichero PDF por cada tag, como antes ---
for keyword in reward_keywords:
    matching_tags = [t for t in tags if keyword.lower() in t.lower()]
    if not matching_tags:
        print("Aviso: no se ha encontrado ningun tag que contenga '%s'" % keyword)
        continue
    for tag in matching_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        fig = plt.figure()
        plt.plot(steps, values, '-')
        plt.xlabel('Training step')
        plt.ylabel(tag)
        plt.grid()
        safe_name = tag.replace('/', '_')
        out_path = os.path.join(out_folder, safe_name + '.pdf')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Guardado: " + out_path)

#--- Perdidas (policy loss, value loss, entropia, etc.): todas juntas en un solo fichero ---
plot_tags = []
for keyword in loss_keywords:
    matching_tags = [t for t in tags if keyword.lower() in t.lower()]
    if not matching_tags:
        print("Aviso: no se ha encontrado ningun tag que contenga '%s'" % keyword)
        continue
    for tag in matching_tags:
        if tag not in plot_tags:
            plot_tags.append(tag)

if not plot_tags:
    print("\nNo se ha graficado ningun tag de perdidas. Revisa la lista de tags "
        "disponibles arriba y ajusta --keywords para que coincidan con los "
        "nombres reales (estos dependen de si el log procede de "
        "stable-baselines o de stable-baselines3).")
else:
    #Disposicion de la rejilla de subplots (maximo 2 columnas)
    n = len(plot_tags)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = [axes[i][j] for i in range(nrows) for j in range(ncols)]

    for ax, tag in zip(axes_flat, plot_tags):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        ax.plot(steps, values, '-')
        ax.set_xlabel('Training step')
        ax.set_ylabel(tag)
        ax.grid()

    #Ocultar los subplots sobrantes si la rejilla no se llena del todo
    for ax in axes_flat[n:]:
        ax.axis('off')

    fig.tight_layout()
    out_path = os.path.join(out_folder, 'loss_curves.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Guardado: " + out_path)

print("\nHecho.")
