import os
import platform
import shutil
import sys
import warnings
import numpy as np
from numpy.linalg import norm

# CAMBIO 1: Se elimina la importacion de TensorFlow.
# stable_baselines (v1) dependia de TF1/TF2 como backend.
# stable_baselines3 usa PyTorch, por lo que TensorFlow ya no es necesario.

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# CAMBIO 2: 'gym' -> 'gymnasium'
import gymnasium as gym

# CAMBIO 3: Las politicas ya no se importan como clases independientes.
# En SB (v1) habia que pasar MlpPolicy como clase al modelo.
# En SB3 las politicas se especifican como string: "MlpPolicy", "MlpLstmPolicy", etc.

# CAMBIO 4: Monitor y resultados vienen ahora de stable_baselines3
from stable_baselines3.common.monitor import Monitor
# results_plotter tambien existe en SB3 bajo la misma ruta relativa
from stable_baselines3.common import results_plotter

# CAMBIO 5: VecEnv viene de stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# CAMBIO 6: Importacion de algoritmos desde stable_baselines3.
# NOTA: PPO2 de SB1 -> PPO en SB3 (se unificaron PPO1 y PPO2).
#       TRPO ya no esta en SB3 base; se puede instalar con sb3-contrib.
#       DDPG si esta disponible en SB3.
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
# TRPO requiere: pip install sb3-contrib
try:
    from sb3_contrib import TRPO
    _TRPO_AVAILABLE = True
except ImportError:
    _TRPO_AVAILABLE = False

# CAMBIO 7: EvalCallback viene de stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback

# Los modulos personalizados no cambian (son independientes de SB/gym)
from custom_modules.custom_policies import CustomPolicy_2x32, CustomPolicy_3x64, CustomPolicy_4x128, \
    CustomLSTMPolicy, CustomPolicy_2x81, CustomPolicy_3_var
from custom_modules.learning_schedules import linear_schedule
from custom_modules.env_fun import make_env
from custom_modules.plot_results import plot_results
from custom_modules.set_axes_equal_3d import set_axes_equal
import argparse
import time
import gym_rendezvous

# CAMBIO 8: Se eliminan los filtros de warnings de TensorFlow (ya no aplica)
warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')

if __name__ == '__main__':

    #Input data
    postprocess = True
    MonteCarlo = True
    tensorboard = True  #necesario para poder extraer luego las curvas de reward, policy loss, value loss y entropia
    eval_environment = True

    #Input settings file
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default="settingsRL.txt",
        help='Input settings file')
    parser.add_argument('--envs', type=int, default=-1,
        help='Number of environments')
    parser.add_argument('--input_model_folder', type=str, default="sol_saved/-1traj_8nb_500epochs_95frac_1/",
        help='Folder of the input BC model to load')
    args = parser.parse_args()
    settings_file = "./settings_files/" + args.settings
    input_model_folder = "./" + args.input_model_folder
    if os.path.isfile(input_model_folder + "best_model.zip"):
        input_model = input_model_folder + "best_model"
    else:
        input_model = input_model_folder + "final_model"

    #Read settings and assign environment and model parameters
    with open(settings_file, "r") as input_file:
        input_file_all = input_file.readlines()
        for line in input_file_all:
            line = line.split()
            if (len(line) > 2):
                globals()[line[0]] = line[1:]
            else:
                globals()[line[0]] = line[1]

    #Settings
    load_model = bool(int(load_model))

    #Environment parameters
    obs_type = int(obs_type)
    random_obs = bool(int(random_obs))
    randomIC = bool(int(randomIC))
    stochastic = bool(int(stochastic))
    termination = bool(int(termination))
    NSTEPS = int(NSTEPS)
    if isinstance(eps_schedule, list):
        eps_schedule = [float(i) for i in eps_schedule]
    else:
        eps_schedule = [float(eps_schedule)]
    lambda_term = float(lambda_term)
    lambda_los = float(lambda_los)
    dr0_max = [float(i) for i in dr0_max]
    dv0_max = [float(i) for i in dv0_max]
    sigma_r = float(sigma_r)
    sigma_v = float(sigma_v)
    sigma_u_rot = float(sigma_u_rot)
    sigma_u_norm = float(sigma_u_norm)
    MTE = bool(int(MTE))
    pr_MTE = float(pr_MTE)
    max_MTE = int(max_MTE)
    acc_max = [float(i) for i in acc_max]
    seed = 0

    #Model parameters
    # CAMBIO 9: En SB3 la politica se pasa como string, no como clase.
    # Por lo tanto, cuando load_model es True, 'policy' debe ser un string
    # como "MlpPolicy" leido directamente desde el fichero de settings.
    # La linea: policy = globals()[policy]  ya NO es necesaria en SB3.
    # Si tu fichero de settings tiene policy = CustomPolicy_2x32, deberas
    # adaptar el entorno o usar net_arch en policy_kwargs (ver comentario mas abajo).
    num_cpu = int(num_cpu)
    if (args.envs > 0):
        num_cpu = int(args.envs)
    learning_rate_in = float(learning_rate_in)
    if (algorithm == "PPO"):
        clip_range_in = float(clip_range_in)
    if learning_rate == "lin":
        learning_rate_type = "linear"
        learning_rate = linear_schedule(learning_rate_in)
    elif learning_rate == "const":
        learning_rate_type = "constant"
        learning_rate = learning_rate_in
    if (algorithm == "PPO"):
        if clip_range == "lin":
            clip_range = linear_schedule(clip_range_in)
        elif clip_range == "const":
            clip_range = clip_range_in
    if (algorithm == "PPO" or algorithm == "A2C" or
        algorithm == "SAC" or algorithm == "TRPO"):
        ent_coef = float(ent_coef)
    gamma = float(gamma)
    if (algorithm == "PPO" or algorithm == "TRPO"):
        lam = float(lam)
    if (algorithm == "PPO"):
        noptepochs = int(noptepochs)
    nminibatches = int(nminibatches)
    niter = int(float(niter))

    # CAMBIO 10: En SB3 PPO, n_steps es el numero de pasos por entorno por update,
    # NO el batch size total. El batch size efectivo = n_steps * num_cpu.
    # En SB (v1) PPO2: n_steps = NSTEPS * nminibatches (batch total).
    # En SB3 PPO:      n_steps = NSTEPS (pasos por env), batch_size = NSTEPS * nminibatches / nminibatches = NSTEPS
    # Se mantiene n_steps igual para preservar el comportamiento de rollout.
    n_steps = int(NSTEPS)
    # CAMBIO 11: batch_size reemplaza a nminibatches en SB3.
    # En SB3, batch_size es el numero de muestras por mini-batch durante el entrenamiento.
    batch_size = int(NSTEPS*num_cpu // nminibatches)  # batch_size total por update
    niter_per_cpu = niter / num_cpu

    #Output folders and log files
    if load_model:
        out_folder_root = input_model_folder + "RL/"
    else:
        out_folder_root = "./sol_saved/"
    n_sol = 1
    out_folder_root += str(num_cpu) + "env_" + \
        str(nminibatches) + "nb_" + \
        str(int(niter/1e6)) + "Mstep_"
    out_folder = out_folder_root + str(n_sol) + "/"
    while os.path.exists(out_folder):
        n_sol += 1
        out_folder = out_folder_root + str(n_sol) + "/"
    os.makedirs(out_folder, exist_ok=True)
    trained_model_name = "final_model"
    if tensorboard:
        tensorboard_log = out_folder
    else:
        tensorboard_log = None
    trained_model_log = out_folder + trained_model_name
    monitor_folder = out_folder + algorithm + "/"
    shutil.copy(settings_file, out_folder)
    os.rename(out_folder + args.settings, out_folder + "settings.txt")

    #Problem data
    Dvmax = float(Dvmax)
    omega = float(omega)
    rKOZ = float(rKOZ)
    beta_cone = float(beta_cone) * np.pi / 180.

    #Read Mission file
    t_nom = []; rx_nom = []; ry_nom = []; rz_nom = []
    vx_nom = []; vy_nom = []; vz_nom = []
    mission_folder = "missions/"
    mission_file = mission_folder + mission_name + ".dat"
    with open(mission_file, "r") as f:
        f.readline()
        for line in f.readlines():
            line = line.split()
            state = np.array(line).astype(np.float64)
            t_nom.append(state[0]); rx_nom.append(state[1]); ry_nom.append(state[2])
            rz_nom.append(state[3]); vx_nom.append(state[4]); vy_nom.append(state[5])
            vz_nom.append(state[6])

    tf = t_nom[-1] - t_nom[0]
    r0 = [rx_nom[0], ry_nom[0], rz_nom[0]]
    v0 = [vx_nom[0], vy_nom[0], vz_nom[0]]
    rTf = [rx_nom[-1], ry_nom[-1], rz_nom[-1]]
    vTf = [vx_nom[-1], vy_nom[-1], vz_nom[-1]]

    # CAMBIO 12: make_env debe devolver una funcion lambda que crea el entorno
    # compatible con Gymnasium (gym.Env con la nueva API step que devuelve
    # 5 valores: obs, reward, terminated, truncated, info).
    # Ver nota en make_env de custom_modules/env_fun.py.
    for rank in range(num_cpu):
        os.makedirs(monitor_folder + "env_" + str(rank) + "/", exist_ok=True)

    env_kwargs = dict(
        obs_type=obs_type, random_obs=random_obs, randomIC=randomIC,
        stochastic=stochastic, termination=termination, NSTEPS=NSTEPS,
        NITER=niter_per_cpu, eps_schedule=eps_schedule, lambda_term=lambda_term,
        Dvmax=Dvmax, tf=tf, omega=omega, rKOZ=rKOZ, beta_cone=beta_cone,
        r0=r0, v0=v0, rTf=rTf, vTf=vTf, dr0_max=dr0_max, dv0_max=dv0_max,
        sigma_r=sigma_r, sigma_v=sigma_v, sigma_u_rot=sigma_u_rot,
        sigma_u_norm=sigma_u_norm, MTE=MTE, pr_MTE=pr_MTE, max_MTE=max_MTE,
        acc_max=acc_max
    )

    if num_cpu <= 1:
        env = DummyVecEnv([make_env(env_id=env_name, rank=rank, seed=seed,
            filename=monitor_folder + "env_" + str(rank) + "/",
            **env_kwargs) for rank in range(num_cpu)])
    else:

        env = SubprocVecEnv([make_env(env_id=env_name, rank=rank, seed=seed,
            filename=monitor_folder + "env_" + str(rank) + "/",
            **env_kwargs) for rank in range(num_cpu)], start_method='spawn')

    if eval_environment:
        eps_schedule_eval = [eps_schedule[-1]]
        eval_env_kwargs = {**env_kwargs, 'NITER': niter_per_cpu / nminibatches,
                          'eps_schedule': eps_schedule_eval}
        eval_env = DummyVecEnv([make_env(env_id=env_name, rank=0, seed=100,
            **eval_env_kwargs)])
        n_eval_episodes = 100
        eval_freq = int(400. * n_steps)
        if not stochastic and not random_obs and not randomIC:
            n_eval_episodes = 1
            eval_freq = int(4. * n_steps)
        eval_callback = EvalCallback(eval_env, n_eval_episodes=n_eval_episodes,
                                     best_model_save_path=out_folder,
                                     log_path=out_folder, eval_freq=eval_freq,
                                     deterministic=True)

    # CAMBIO 14: Creacion de modelos con la API de SB3.
    # Diferencias clave por algoritmo:
    #
    # PPO2 (SB1) -> PPO (SB3):
    #   - nminibatches -> batch_size
    #   - noptepochs  -> n_epochs
    #   - cliprange   -> clip_range
    #   - cliprange_vf -> clip_range_vf (en SB3 se puede pasar None para usar el mismo que clip_range)
    #   - lam         -> gae_lambda
    #
    # A2C (SB1->SB3): API muy similar, lr_schedule desaparece (usar learning_rate callable)
    #
    # DDPG (SB1->SB3): nb_train_steps desaparece; se controla con learning_starts y train_freq
    #
    # TRPO: no esta en SB3 base, usar sb3-contrib
    
    # CAMBIO 15: En SB3, .load() no acepta parametros de arquitectura distintos
    # a los guardados. Para cambiar hiperparametros al cargar, usese set_parameters()
    # o reconstruye el modelo y carga solo los pesos con model.set_parameters().

    if algorithm == "PPO":
        ppo_kwargs = dict(
            policy="MlpPolicy",       # string en SB3, no clase
            env=env,
            n_steps=n_steps,
            batch_size=batch_size,    # CAMBIO: nminibatches -> batch_size
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range_vf=None,       # CAMBIO: cliprange_vf=-1 -> None (mismo que clip_range)
            gae_lambda=lam,           # CAMBIO: lam -> gae_lambda
            n_epochs=noptepochs,      # CAMBIO: noptepochs -> n_epochs
            learning_rate=learning_rate,
            clip_range=clip_range,
            tensorboard_log=tensorboard_log,
            verbose=1
        )
        if not load_model:
            model = PPO(**ppo_kwargs)
        else:
            # CAMBIO 16: En SB3, .load() carga arquitectura + pesos del fichero.
            # Para continuar entrenando hay que pasar env= de nuevo.
            model = PPO.load(input_model, env=env, **{k: v for k, v in ppo_kwargs.items()
                             if k not in ('policy', 'env')})
            model.tensorboard_log = tensorboard_log

    elif algorithm == "A2C":
        model = A2C(
            policy="MlpPolicy",
            env=env,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            learning_rate=learning_rate_in,
            # CAMBIO: lr_schedule ya no existe; pasar learning_rate como callable
            # si se quiere schedule lineal (ya manejado por linear_schedule).
            tensorboard_log=tensorboard_log,
            verbose=1
        )

    elif algorithm == "DDPG":
        model = DDPG(
            policy="MlpPolicy",
            env=env,
            gamma=gamma,
            # CAMBIO: actor_lr/critic_lr -> learning_rate (una sola LR en SB3 DDPG)
            learning_rate=learning_rate_in,
            # CAMBIO: nb_train_steps eliminado; controlado por train_freq y gradient_steps
            tensorboard_log=tensorboard_log,
            verbose=1
        )

    elif algorithm == "SAC":
        model = SAC(
            policy="MlpPolicy",
            env=env,
            gamma=gamma,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            batch_size=NSTEPS,
            tensorboard_log=tensorboard_log,
            verbose=1
        )

    elif algorithm == "TD3":
        model = TD3(
            policy="MlpPolicy",
            env=env,
            gamma=gamma,
            batch_size=NSTEPS,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            verbose=1
        )

    elif algorithm == "TRPO":
        if not _TRPO_AVAILABLE:
            raise ImportError("TRPO requiere sb3-contrib: pip install sb3-contrib")
        model = TRPO(
            policy="MlpPolicy",
            env=env,
            gamma=gamma,
            # CAMBIO: timesteps_per_batch -> n_steps; entcoeff -> ent_coef
            n_steps=NSTEPS,
            gae_lambda=lam,
            ent_coef=ent_coef,
            tensorboard_log=tensorboard_log,
            verbose=1
        )

    # El loop de entrenamiento no cambia
    start_time = time.time()
    if eval_environment:
        model.learn(total_timesteps=niter, callback=eval_callback)
    else:
        model.learn(total_timesteps=niter)
    end_time = time.time()

    model.save(trained_model_log)
    print("End Training.")

    f_out_time = open(out_folder + "time.txt", "w")
    if eval_environment:
        f_out_time.write("%20s\t%20s\n" % ("# elapsed time [s]", "best J"))
        f_out_time.write("%20.7f\t%20.7f\n" % (end_time - start_time, eval_callback.best_mean_reward))
    else:
        f_out_time.write("%20s\n" % ("# elapsed time [s]"))
        f_out_time.write("%20.7f\n" % (end_time - start_time))
    f_out_time.close()

    if postprocess:
        print("Post-processing\n")
        os.system('python main_rendezvous_load.py --folder ' + out_folder)
        if MonteCarlo and (stochastic or random_obs or randomIC):
            print("Monte Carlo\n")
            os.system('python main_rendezvous_MC.py --folder ' + out_folder)
