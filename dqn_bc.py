
"""
Se define la clase del modelo de Deep Q-Network (DQN)

"""
###################### Librerias
import time
from collections import deque, namedtuple

import gym
import numpy as np
import tensorflow as tf
import utils

#Construccion de modelo neuronal
from tensorflow.keras import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Input #type: ignore
from tensorflow.keras.losses import MSE #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore


from utils import update_target_network, check_update_conditions, get_experiences
from statistics import mean 
from datetime import datetime

###################### Definición de la clase DQN_BC

class DQN_BC():



    def __init__(self,

    env, #Environment
    learning_rate,  #Tasa de aprendizaje para el entrenamiento de la red
    gamma, #"Discount Factor"
    num_episodes, #Número de episodios
    max_num_timesteps, #Número maximo de timesteps
    memory_size, #Tamaño de memoria del buffer
    state_size, #Numero de variables de estado
    num_actions, #Numero de acciones
    q_network, #Modelo Keras para Q-values
    target_q_network #Modelo Keras para los targets

    
    ):
        
        self.env = env
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.num_actions = num_actions

        self.q_network = q_network
        self.target_q_network = target_q_network

        self.gamma = gamma
        self.num_episodes = num_episodes
        self.max_num_timesteps = max_num_timesteps
        self.memory_size = memory_size


    def compute_loss(self, experiences):
        """ 
        Calcula la funcion de pérdida para el entrenamiento de la red
        
        Args:
        experiences: (tuple) tupla de  ["state", "action", "reward", "next_state"] namedtuples
        gamma: (float) "discount factor".
        q_network: (tf.keras.Sequential) modelo Keras para predecir q_values
        target_q_network: (tf.keras.Sequential) modelo Keras para predecir los targets
            
        Returns:
        loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) Error Cuadrático Medio entre
                los targets y los Q-values.
        """
        
        # Se desempacan las variables de la tupla de experiencias.
        states, actions, rewards, next_states, done_vals = experiences
        
        # Calculo del Q^(s´,a´) max.
        max_qsa = tf.reduce_max(self.target_q_network(next_states), axis=-1)
        
    
        y_targets = rewards+(1-done_vals)*self.gamma*max_qsa
        
        # Se obtienen los q_values.
        q_values = self.q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))
            
        # Compute the loss.
        loss = MSE(y_targets, q_values) 
        
        return loss


    @tf.function
    def agent_learning(self, experiences):
        """
        Actualiza los pess de las redes Q networks.
        
        Args:
        experiences: (tuple) tupla de ["state", "action", "reward", "next_state", "done"] 
        gamma: (float) "discount factor".
        
        """
        
        # Se calcula la pérdida o error (loss).
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences, self.gamma, self.q_network, self.target_q_network)

        # Se obtienen los gradientes del error con respecto a los pesos de la red
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        
        # Se actualizan los pesos de q_network.
        optimizer = Adam(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # Se actualizan los pesos de target q_network.
        update_target_network(self.q_network, self.target_q_network) 
    


    def train(self):
        """
        Proceso de entrenamiento del agente DQN.
        
        Args:
        
        
        """
        memory_buffer = deque(maxlen=self.memory_size)
        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        q_val_list = []               
        eps_avg_max_q_val = []
        # Set the target network weights equal to the Q-Network weights.
        self.target_q_network.set_weights(self.q_network.get_weights())


        for i in range(self.num_episodes):
    
            start = time.time()

            for t in range(self.max_num_timesteps):
                
                state = self.env.reset()

                #Almacenamiento de nuevo estado, accion, recompesa en el buffer
                next_state, action, reward, done, _ = self.env.step()
                memory_buffer.append(experience(state, action, reward, next_state, done))
                
                # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
                update = check_update_conditions(t, self.num_step_for_update, memory_buffer)
                
                if update:
                    # Sample random mini-batch of experience tuples (S,A,R,S') from D.
                    experiences = get_experiences(memory_buffer)
                    
                    # Set the y targets, perform a gradient descent step,
                    # and update the network weights.
                    self.agent_learning(experiences, self.gamma)

                    

                state_qn = np.expand_dims(state, axis=0)  #Se modifica el shape para la red neuronal
                    
                q_val_list.append(
                                    tf.math.reduce_mean(
                                        
                                        self.q_network(state_qn)
                
                                    ).numpy()
                                
                                )
                if done:
                    break

            eps_avg_max_q_val.append(mean(q_val_list))
            
            fin = time.time()
            hora = str(datetime.now().strftime("%H:%M:%S"))
            average_report = round(mean(q_val_list),4)
            time_report = round((fin- start)/60,4)
            print(f"Iteracion : {i} ----- Avg Q-value : {average_report} ----- Duración de iteración : {time_report} mins ----- Hora: {hora}")
            q_val_list = []

            
        return eps_avg_max_q_val



                            


            





        