
from gym import Env 
from gym.spaces import Discrete, Box, MultiBinary
import numpy as np
import random





class CustomerEnv(Env):
    
    def __init__(self,
                data_states_past, #Dataframe con la descripción de estado en t-1
                data_states_act, #Dataframe con la descripción de estado en t
                data_acns_rwrd, #Dataframe con la descripción de acciones y rewards
                observation_list, #Lista con el nombre de las variables que describen el estado
                acn_list, #Lista con el nombre de las acciones 
                minmaxscaler #Funcion para normalizar los datos
                ):
        #Se asignan variables
        self.observation_list = observation_list
        
        self.data_states_past = data_states_past
        self.minmaxscaler = minmaxscaler
        self.acn_list = acn_list
        self.data_states_act = data_states_act
        self.data_acns_rwrd = data_acns_rwrd

        #Obtención del estado del cliente inicial (aleatorio)
        #se selecciona de manera aleatoria un cliente
        self.cliente_random = random.choice(list(self.data_states_past.id.unique()))
        #se selecciona de manera aleatoria un periodo de tiempo
        self.periodo_temp = random.choice(list(self.data_states_past[self.data_states_past["id"] == self.cliente_random].periodo.unique()))
        #Se obtiene el estado del cliente aleatorio en el periodo de tiempo aleatorio
        self.state = self.data_states_past.\
                                            loc[((self.data_states_past["id"] == self.cliente_random) &
                                                    (self.data_states_past["periodo"] == self.periodo_temp)), self.observation_list]
        self.state = self.minmaxscaler.\
                                        transform(self.state).\
                                        reshape(len(self.observation_list))
        

        #Asignación de espacio de acciones como Discreto
        self.action_space = Discrete(len(self.acn_list))
        
        
        #Se inicializa contador en 0
        self.contador = 0


    #Definicion de un paso (teniendo en cuenta el estado, se toma la accion y se recibe la recompensa)
    def step(self):

        #Se incrementa en 1 el mes y contador
        self.periodo_temp += 1
        self.contador += 1

        #Condicional para ajustar el periodo_temp
        if self.periodo_temp == 202113:
            self.periodo_temp = 202201
        else:
            self.periodo_temp = self.periodo_temp

        #Se obtiene la accion valida para ese periodo de tiempo mediante una seleccion aleatoria
        self.action =  random.choice(self.data_acns_rwrd[(self.data_acns_rwrd["id"] == self.cliente_random) &
                                        (self.data_acns_rwrd["periodo"] == self.periodo_temp)].num_accion.unique())

        #Se obtiene el estado actual
        self.state = self.data_states_act.loc[((self.data_states_act["periodo"] == self.periodo_temp) & 
                                                (self.data_states_act["id"] == self.cliente_random)), self.observation_list]
        self.state = self.minmaxscaler.\
                                        transform(self.state).\
                                        reshape(len(self.observation_list))
        
        #Se obtiene la recompensa despues de tomar la accion
        reward = self.data_acns_rwrd[self.data_acns_rwrd["num_accion"] == self.action].reward.values[0]
        done = False
        if self.contador == 1:
            done = True
        else:
            done = False
        info = {}

        #retorna la informacion
        return self.state, self.action, reward, done, info

    def reset(self):
        #Obtención del estado del cliente inicial (aleatorio)
        #se selecciona de manera aleatoria un cliente
        self.cliente_random = random.choice(list(self.data_states_past.id.unique()))
        #se selecciona de manera aleatoria un periodo de tiempo
        self.periodo_temp = random.choice(list(self.data_states_past[self.data_states_past["id"] == self.cliente_random].periodo.unique()))

        #Se obtiene el estado del cliente aleatorio
        self.state = self.data_states_past.\
                                            loc[((self.data_states_past["id"] == self.cliente_random) &
                                                    (self.data_states_past["periodo"] == self.periodo_temp)), self.observation_list]
        self.state = self.minmaxscaler.\
                                        transform(self.state).\
                                        reshape(len(self.observation_list))

        return self.state
    def  render(self):
        pass