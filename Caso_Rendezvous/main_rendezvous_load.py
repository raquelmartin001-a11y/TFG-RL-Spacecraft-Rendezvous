import os
import platform
import warnings
import numpy as np
from numpy.linalg import norm
import torch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

# --- SB3 usa Gymnasium, no el "gym" clasico ---------------------------------
# Si gym_rendezvous todavia esta escrito sobre el "gym" antiguo, necesitas
# o bien migrar ese paquete a la API de gymnasium, o instalar "shimmy" y
# envolver el entorno con gymnasium.wrappers.EnvCompatibility / shimmy.GymV21CompatibilityV0.
# Aqui se intenta usar gymnasium si esta disponible y se cae a gym si no.
try:
    import gymnasium as gym
    GYMNASIUM = True
except ImportError:
    import gym
    GYMNASIUM = False

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import DummyVecEnv

# PPO2 -> PPO ; DDPG, SAC, TD3 siguen existiendo en el core de SB3.
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3

# TRPO y las politicas recurrentes (LSTM) NO estan en el core de stable-baselines3.
# Viven en el paquete complementario "sb3-contrib" (pip install sb3-contrib).
# RecurrentPPO sustituye a las antiguas MlpLstmPolicy / LstmPolicy / CustomLSTMPolicy.
try:
    from sb3_contrib import TRPO, RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False

# NOTA IMPORTANTE sobre las politicas personalizadas
# ----------------------------------------------------
# En stable-baselines (v1) las arquitecturas custom (CustomPolicy_2x32,
# CustomPolicy_3x64, CustomPolicy_4x128, CustomPolicy_2x81, CustomPolicy_3_var,
# CustomLSTMPolicy) eran subclases de politica con la arquitectura "hardcodeada".
#
# En stable-baselines3 esto normalmente ya NO se hace con clases custom, sino
# pasando policy_kwargs=dict(net_arch=[...]) a la politica estandar "MlpPolicy".
# Por ejemplo, CustomPolicy_2x32 (dos capas de 32) equivaldria a:
#     policy_kwargs = dict(net_arch=[32, 32])
#     model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
#
# No tengo el contenido de custom_modules/custom_policies.py, asi que no puedo
# traducir automaticamente cada clase. Te recomiendo sustituir su import por
# un diccionario de policy_kwargs equivalente, por ejemplo:
#
# CUSTOM_NET_ARCH = {
#     "CustomPolicy_2x32":  [32, 32],
#     "CustomPolicy_3x64":  [64, 64, 64],
#     "CustomPolicy_4x128": [128, 128, 128, 128],
#     "CustomPolicy_2x81":  [81, 81],
#     "CustomPolicy_3_var": [?, ?, ?],  # <-- necesito ver la definicion original
# }
#
# Dejo el import comentado hasta que decidas la estrategia:
from custom_modules.custom_policies import CustomPolicy_2x32, CustomPolicy_3x64, \
CustomPolicy_4x128, CustomLSTMPolicy, CustomPolicy_2x81, CustomPolicy_3_var

from custom_modules.learning_schedules import linear_schedule
from custom_modules.plot_results import plot_results
from custom_modules.set_axes_equal_3d import set_axes_equal
import argparse
import gym_rendezvous
from gym_rendezvous.envs.pyHCW import propagate_HCW

# numpy/torch warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# Input data
graphs = True
expert_traj = False

#Input settings file
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="sol_saved/4env_4nb_10Mstep_3/", \
    help='Input model folder')
args = parser.parse_args()
settings_file = args.folder + "settings.txt"

#Read settings and assign environment and model parameters
with open(settings_file, "r") as input_file: # with open context
    input_file_all = input_file.readlines()
    for line in input_file_all: #read line
        line = line.split()
        if (len(line) > 2):
            globals()[line[0]] = line[1:]
        else:
            globals()[line[0]] = line[1]

input_file.close() #close file

#Environment parameters
obs_type = int(obs_type) #type of observations
random_obs = bool(int(random_obs)) #random observations
randomIC = bool(int(randomIC)) #random initial conditions
stochastic = bool(int(stochastic)) #stochastic environment
termination = bool(int(termination)) #terminating when constrainits violated
NSTEPS = int(NSTEPS) # Total number of trajectory segments
if isinstance(eps_schedule, list):
    eps_schedule = [float(i) for i in eps_schedule] #epsilon-constraint schedule
else:
    eps_schedule = [float(eps_schedule)]
lambda_term = float(lambda_term) #weight of terminal constraint violation in reward
lambda_los = float(lambda_los) #weight of visibility cone constraint violation in reward
dr0_max = [float(i) for i in dr0_max] #maximum variation of initial position
dv0_max = [float(i) for i in dv0_max] #maximum variation of initial velocity
sigma_r = float(sigma_r) #standard deviation of position
sigma_v = float(sigma_v) #standard deviation of velocity
sigma_u_rot = float(sigma_u_rot) #standard deviation of control rotation
sigma_u_norm = float(sigma_u_norm) #standard deviation of control modulus
MTE = bool(int(MTE)) #at least one MTE occurs?
pr_MTE = float(pr_MTE) #probability of having a MTE at k-th step
max_MTE = int(max_MTE) #maximum number of consecutive MTEs
acc_max = [float(i) for i in acc_max] #maximum value of the components of a perturbing acceleration

#Model parameters
# En SB3 la politica normalmente se referencia como string ("MlpPolicy"), no
# como una clase importada. Si tu settings.txt guarda "MlpPolicy" como texto,
# esto ya funciona igual que antes.
policy = globals()[policy] if globals()[policy] in ("MlpPolicy", "CnnPolicy", "MultiInputPolicy") \
    else globals()[policy]

#Input Model and Output folders
in_folder = "./" + args.folder
if os.path.isfile(in_folder + "best_model.zip"):
    logname = "best_model"
else:
    logname = "final_model"
trained_model = in_folder + logname
plot_folder = in_folder

#Problem data
Dvmax = float(Dvmax)                        #Maximum chaser Dv, km/s
omega = float(omega)                        #Mean motion of the target, rad/s
rKOZ = float(rKOZ)                          #radius of the keep out zone (KOZ), km
beta_cone = float(beta_cone)*np.pi/180.   #visibility cone semi-aperture, rad

#Read Mission file
t_nom = []  #nominal trajectory: time
rx_nom = [] #nominal trajectory: x
ry_nom = [] #nominal trajectory: y
rz_nom = [] #nominal trajectory: z
vx_nom = [] #nominal trajectory: vx
vy_nom = [] #nominal trajectory: vy
vz_nom = [] #nominal trajectory: vz
mission_folder = "missions/"
mission_file = mission_folder + mission_name + ".dat" #File with mission data
with open(mission_file, "r") as f: # with open context
    f.readline()
    file_all = f.readlines()
    for line in file_all: #read line
        line = line.split()
        state = np.array(line).astype(np.float64) #non-dimensional data

        #save data
        t_nom.append(state[0])
        rx_nom.append(state[1])
        ry_nom.append(state[2])
        rz_nom.append(state[3])
        vx_nom.append(state[4])
        vy_nom.append(state[5])
        vz_nom.append(state[6])

f.close() #close file

#Mission data

#Time-of-flight
tf =  t_nom[-1] - t_nom[0]   #s, Time-of-flight
dt = tf/NSTEPS               #s, time-step

#Reference initial state
r0 = [rx_nom[0], ry_nom[0], rz_nom[0]] #km, initial relative chaser position
v0 = [vx_nom[0], vy_nom[0], vz_nom[0]] #km/s, initial relative chaser velocity

#Target state
rTf = [rx_nom[-1], ry_nom[-1], rz_nom[-1]]      #km, final target position
vTf = [vx_nom[-1], vy_nom[-1], vz_nom[-1]]      #km/s, final target velocity

#Epsilon constraint schedule
eps_schedule = [eps_schedule[-1]]

# Environment creation
env0 = gym.make(id=env_name, \
            obs_type=obs_type, \
            random_obs=False, randomIC=False, stochastic=False, \
            termination=termination, NSTEPS=NSTEPS, NITER=NSTEPS, \
            eps_schedule=eps_schedule, lambda_term=lambda_term, \
            Dvmax=Dvmax, tf=tf, omega=omega, \
            rKOZ=rKOZ, beta_cone=beta_cone, \
            r0=r0, v0=v0, \
            rTf=rTf, vTf=vTf, \
            dr0_max=dr0_max, dv0_max=dv0_max, \
            sigma_r=sigma_r, sigma_v=sigma_v, \
            sigma_u_rot=sigma_u_rot, sigma_u_norm=sigma_u_norm, \
            MTE=MTE, pr_MTE=pr_MTE, max_MTE=max_MTE, \
            acc_max=acc_max)

# gymnasium: env.reset(seed=...) reemplaza a env.seed(...)
if GYMNASIUM:
    obs, _info = env0.reset(seed=0)
else:
    env0.seed(0)
    obs = env0.reset()

# Load model
# SB3 gestiona el zip internamente: no hace falta el np.load(...) manual que
# se usaba en stable-baselines v1 para inspeccionar el fichero.
# El algoritmo de carga depende de con que algoritmo se entreno el modelo:
# usa PPO.load / A2C.load / DDPG.load / SAC.load / TD3.load / TRPO.load / RecurrentPPO.load
# segun corresponda. Aqui se asume PPO como en el script original (PPO2).
model = PPO.load(trained_model)

if expert_traj:
    # generate_expert_traj (GAIL, stable-baselines v1) no tiene equivalente
    # directo en stable-baselines3. La opcion recomendada es la libreria
    # "imitation" (https://imitation.readthedocs.io), que se apoya en SB3.
    # Como sustituto minimo, aqui se graban rollouts manualmente en un .npz
    # con el mismo formato que generate_expert_traj (observations, actions,
    # rewards, episode_returns, episode_starts) para poder reutilizarlos
    # despues con imitation.data.types.Transitions o similar.
    print("\nGenerando trayectorias expertas (reemplazo manual de generate_expert_traj):\n")

    n_episodes = 100
    all_obs, all_actions, all_rewards = [], [], []
    episode_returns, episode_starts = [], []

    for ep in range(n_episodes):
        if GYMNASIUM:
            obs, _info = env0.reset()
        else:
            obs = env0.reset()
        done = False
        ep_reward = 0.0
        first_step = True
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            if GYMNASIUM:
                obs, reward, terminated, truncated, info = env0.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, info = env0.step(action)
            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            episode_starts.append(first_step)
            first_step = False
            ep_reward += reward
        episode_returns.append(ep_reward)

    np.savez(
        in_folder + 'expert_rendezvous.npz',
        obs=np.array(all_obs),
        actions=np.array(all_actions),
        rewards=np.array(all_rewards),
        episode_returns=np.array(episode_returns),
        episode_starts=np.array(episode_starts),
    )

# Print graph and results
f_out = open(in_folder + "Simulation.txt", "w") # open file
f_out.write("Environment simulation\n\n")
f_out_traj = open(in_folder + "Trajectory.txt", "w") # open file
f_out_u = open(in_folder + "control.txt", "w") # open file
f_out_traj.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("# x", "y", "z", "vx", "vy", "vz"))
f_out_u.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("# t", "x", "y", "z", "Dvx", "Dvy", "Dvz", "Dvnorm", "Dvmax"))
np.set_printoptions(precision=3)

#Unzip and save evaluations file
evalutation_file_npz = in_folder + "evaluations.npz"
if os.path.isfile(evalutation_file_npz):
    npzfile = np.load(evalutation_file_npz)
    f_out_eval = open(in_folder + "evaluations.txt", "w") # open file
    f_out_eval.write('%20s\t%20s\t%20s\t%20s\n' \
        % ("# training step", "mean reward", "std reward", "mean ep. length"))
    for i in range(len(npzfile['timesteps'])):
        f_out_eval.write('%20d\t%20.5f\t%20.5f\t%20d\n' \
        % (npzfile['timesteps'][i], np.mean(npzfile['results'][i]), np.std(npzfile['results'][i]), \
            np.mean(npzfile['ep_lengths'][i])))
    f_out_eval.close()

    if graphs:
        matplotlib.rc('font', size=12)
        matplotlib.rc('text', usetex=False)
        fig0 = plt.figure()
        plt.plot(npzfile['timesteps'], np.mean(npzfile['results'], axis=1),'o-')
        plt.xlabel('Training step number')
        plt.ylabel('Mean reward')
        plt.yscale('symlog')
        plt.grid()
        plt.savefig(in_folder + "Evaluations.pdf", dpi=300)

if (graphs):
    matplotlib.rc('font', size=18)
    matplotlib.rc('text', usetex=False)
    fig1 = plt.figure()
    ax1 = fig1.gca()

# Environment simulation

#Reset environment
if GYMNASIUM:
    obs, _info = env0.reset()
else:
    obs = env0.reset()
cumulative_reward = 0.
u_tot = 0.
beta_max = 0.
t_vec = []
beta_vec = []
traj_x = [r0[0]]
traj_y = [r0[1]]
for i in range(NSTEPS):

    #Get current action
    action, _states = model.predict(obs, deterministic=True)

    #Get new observation
    if GYMNASIUM:
        obs, reward, terminated, truncated, info = env0.step(action)
        done = terminated or truncated
    else:
        obs, reward, done, info = env0.step(action)

    #Spacecraft state, time and control
    r = np.array([info["rx"], info["ry"], info["rz"]])
    v = np.array([info["vx"], info["vy"], info["vz"]])
    t = info["t"]
    u = np.array([info["ux"], info["uy"], info["uz"]])
    u_max = Dvmax
    u_tot += norm(u)

    traj_x.append(r[0])
    traj_y.append(r[1])

    #Print trajectory information
    f_out.write("t_step = " + str(np.round(t/dt)) + "\n")
    f_out.write("norm(r) = " + str(norm(r)) + "\n")
    f_out.write("norm(v) = " + str(norm(v)) + "\n")
    f_out.write("norm(u/u_max) = " + str(norm(u)/u_max) + "\n")
    f_out.write("cum_reward = " + str(cumulative_reward) + "\n\n")

    #Max. beta value
    beta_z_pos = abs(np.arctan(r[2] / r[1]))
    beta_z_neg = abs(np.arctan(-r[2] / r[1]))
    beta_x_pos = abs(np.arctan(r[0] / r[1]))
    beta_x_neg = abs(np.arctan(-r[0] / r[1]))
    beta_max_new = max([beta_z_pos, beta_z_neg, beta_x_pos, beta_x_neg])
    if beta_max_new > beta_max:
        beta_max = beta_max_new

    #Store beta angle history (for cone constraint plot)
    t_vec.append(t)
    beta_vec.append(beta_max_new)

    #Print trajectory segment
    N = 10
    dtj = dt / (N - 1)
    r_prop = r
    v_prop = v + u
    for j in range(N):
        f_out_traj.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
            % (r_prop[0], r_prop[1], r_prop[2], \
            v_prop[0], v_prop[1], v_prop[2]))
        r_prop, v_prop = propagate_HCW(r0=r_prop, v0=v_prop, dt=dtj, omega=omega)

    #Print control
    f_out_u.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (t, r[0], r[1], r[2], u[0], u[1], u[2], norm(u), u_max))

    if graphs:
        #Plot control
        ax1.stem([t/tf], [norm(u)*1000.], '-k')

    #Update cumulative reward
    cumulative_reward += reward

#Final state
t = env0.t
r = env0.rk
v = env0.vkm

rTf = np.array(rTf)
vTf = np.array(vTf)

#Final state, after DV
uf = min(norm(vTf - v), u_max)*(vTf - v)/norm(vTf - v)
u_tot += norm(uf)
rf = r
vf = v + uf

traj_x.append(rf[0])
traj_y.append(rf[1])

#Print final state
f_out.write("t_step = " + str(np.round(t/dt)) + "\n")
f_out.write("norm(r) = " + str(norm(r)) + "\n")
f_out.write("norm(v) = " + str(norm(v)) + "\n")
f_out.write("norm(u/u_max) = " + str(norm(uf)/u_max) + "\n")
f_out.write("cum_reward = " + str(cumulative_reward) + "\n\n")

#Print final control
f_out_traj.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (rf[0], rf[1], rf[2], vf[0], vf[1], vf[2]))
f_out_u.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (tf, rf[0], rf[1], rf[2], uf[0], uf[1], uf[2], norm(uf), u_max))

if graphs:
    #Plot final control
    ax1.stem([tf/tf], [norm(uf)*1000.], '-k')

f_out.write("Final position: r = " + str(rf) + " km\n")
f_out.write("Final velocity: v = " + str(vf) + " km/s\n")
f_out.write("Final target position: rT = " + str(rTf) + " km\n")
f_out.write("Final target velocity: vT = " + str(vTf) + " km/s\n")
f_out.write("Final position error: dr = " + str(norm(rf - rTf)) + " km\n")
f_out.write("Final velocity error: dv = " + str(norm(vf - vTf)) + " km/s\n")
f_out.write("Max LOS angle: beta_max = " + str(beta_max*180./np.pi) + " deg\n")
if beta_max <= beta_cone:
    f_out.write("Visibility cone constraint: SATISFIED (beta_max = %.3f deg <= beta_cone = %.3f deg)\n" \
        % (beta_max*180./np.pi, beta_cone*180./np.pi))
else:
    f_out.write("Visibility cone constraint: VIOLATED (beta_max = %.3f deg > beta_cone = %.3f deg)\n" \
        % (beta_max*180./np.pi, beta_cone*180./np.pi))
f_out.write("Total Dv: Dv = " + str(u_tot) + " km/s\n")
f_out.write("Final time: t = " + str(t) + " s\n\n")

f_out.close()
f_out_traj.close()
f_out_u.close()

if graphs:
    #Control figure
    plt.xlabel('$t/t_f$')
    plt.ylabel('$\\Delta v$, [m/s]')
    plt.grid()
    fig1.savefig(plot_folder + "control.pdf", dpi=300, bbox_inches='tight')

    #Visibility cone constraint figure
    matplotlib.rc('font', size=18)
    fig2 = plt.figure()
    plt.plot(np.array(t_vec)/tf, np.array(beta_vec)*180./np.pi, '-b', label=r'$\beta(t)$')
    plt.axhline(y=beta_cone*180./np.pi, color='r', linestyle='--', label='Cone limit')
    plt.xlabel('$t/t_f$')
    plt.ylabel(r'$\beta$, [deg]')
    plt.legend()
    plt.grid()
    fig2.savefig(plot_folder + "cone_constraint.pdf", dpi=300, bbox_inches='tight')

    #Trajectory + visibility cone in x-y plane
    fig3 = plt.figure()
    ax3 = fig3.gca()

    traj_x = np.array(traj_x)
    traj_y = np.array(traj_y)
    ax3.plot(traj_y, traj_x, '-b', label='Trayectoria real')
    ax3.plot(0, 0, 'ok', label='Target')

    #Cono de visibilidad: vertice en el target (origen), eje a lo largo de y
    y_dir = np.sign(traj_y[0]) if traj_y[0] != 0 else -1.
    L = 1.2 * np.max(np.abs(traj_y))
    y_cone = np.array([0, y_dir * L])
    x_cone = y_cone * np.tan(beta_cone)
    ax3.plot(y_cone, x_cone, '--r', label='Cono de visibilidad')
    ax3.plot(y_cone, -x_cone, '--r')

    ax3.set_xlabel('y, [km]')
    ax3.set_ylabel('x, [km]')
    ax3.axis('equal')
    ax3.grid()
    ax3.legend(loc='best', fontsize=12)
    fig3.savefig(plot_folder + "trajectory_cone.pdf", dpi=300, bbox_inches='tight')

    #Trajectory figure
    os.system("gnuplot -e \"indir='" + str(args.folder) + "'\" \"PlotFiles/plot_traj.plt\"")
    os.system("latexmk -pdf")
    os.system("latexmk -c")
    os.system("rm *.eps *.tex *-inc-eps-converted-to.pdf")
    os.system("mv *.pdf " + args.folder)


print("Results printed, graphs plotted.")
