
import pandas as pd
import numpy as np

from dqn_bc import DQN_BC
from customer_environment import CustomerEnv
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from collections import deque, namedtuple

from sparky_bc import Sparky
sp = Sparky('dborras', 'IMPALA_PROD')

##############################################################
########## DEFINICION DE LISTA DE OBSERVACIONES Y ACCIONES
observation_list = [
                    "mes",
                    "estandar_basico",
                    "estandar_medio",
                    "estandar_plus",
                    "potencial_basico",
                    "potencial_medio",
                    "potencial_alto",
                    "preferencial_bc",
                    "preferencial_rel",
                    "preferencial_plus",
                    "cluster_as",
                    "cluster_ac",
                    "cluster_it",
                    "cluster_np",
                    "cluster_bc",
                    "cluster_je",
                    "cluster_pt",
                    "tiene_ahorros",
                    "tiene_corriente",
                    "tiene_afc",
                    "tiene_mastercard",
                    "tiene_visa",
                    "tiene_amex",
                    "tiene_vehi_sufi",
                    "tiene_edu_sufi",
                    "tiene_cdt_banco",
                    "tiene_coberturas",
                    "tiene_sgr_vida",
                    "tiene_sgr_desempleo",
                    "tiene_sgr_salud",
                    "tiene_sgr_hogar",
                    "tiene_sgr_vehiculo",
                    "tiene_fondo_liquidez",
                    "tiene_fondo_renta_fija",
                    "tiene_fondo_balanceado",
                    "tiene_fondo_renta_variable",
                    "tiene_fondo_alternativo",
                    "tiene_adquivivienda",
                    "tiene_reforvivienda",
                    "tiene_libre_inversion",
                    "tiene_libranz_estan",
                    "tiene_libranz_pensi",
                    "tiene_crediagil",
                    "cantidad_portafolio",
                    "trx_compras"                   
                    ]
                
acn_list =   ['acn_aumento_cupo_bpo',
       'acn_crediagil_bpo',
        'acn_libranza_bpo',
         'acn_libre_inversion_bpo',
       'acn_rediferido_bpo',
        'acn_retanqueo_bpo',
         'acn_tdc_bpo']

##############################################################
########## DATAFRAMES


data_state_act =


data_state_menos_1

##############################################################
########## DEFINICION DE PARAMETROS


memory_size = 1000_000
gamma = 0.995
learning_rate = 0.001
num_step_for_update = 4
state_size = len(observation_list)
num_actions = len(acn_list) 
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

minmaxscaler = MinMaxScaler(feature_range = (0,1))
minmaxscaler.fit(pd.concat([data_state_act[observation_list], data_state_menos_1[observation_list]], ignore_index= True))





seed = 0  # Semilla para el generador aleatorio
num_episodes = 5 # Numero de episodios
max_num_timesteps = 10 # numero maximo de timesteps


memory_buffer = deque(maxlen=memory_size)

tf.random.set_seed(seed)
tf.config.run_functions_eagerly(True)



