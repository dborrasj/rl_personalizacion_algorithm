import pandas as pd
import numpy as np

from sparky_bc import Sparky
import re

sp = Sparky('dborras', 'IMPALA_PROD')
################################################################################
####################################    BPO

data_bpo = sp.helper.obtener_dataframe(""" 

SELECT
num_doc,
CAST(YEAR(f_gestionado) AS int)*100 + CAST(MONTH(f_gestionado) AS int) AS periodo,
CAST(MONTH(f_gestionado) AS int) as mes,
CAST(DAY(f_gestionado) AS int) as dia,
prod_ofrecido,
prod_aceptado,
causal_no_venta
FROM resultados_vspc_canales.bpo_gestionados
WHERE desc_contacto LIKE "%ontacto efectivo%" 
        AND (YEAR(f_gestionado)*100 + MONTH(f_gestionado)) BETWEEN 202200 and 202212
        

""")



#Se vuelve minuscula todos los campos
data_bpo["prod_ofrecido"] = data_bpo.prod_ofrecido.str.lower()
data_bpo["prod_aceptado"] = data_bpo.prod_aceptado.str.lower()
data_bpo["causal_no_venta"] = data_bpo.causal_no_venta.str.lower()

#Se cambian los valores vacios y nan de la columna prod_aceptado por "no_acepta_producto"
data_bpo["prod_aceptado"] = data_bpo.prod_aceptado.replace("", "no_acepta_producto")
data_bpo.prod_aceptado.fillna("no_acepta_producto", inplace = True)

#se normaliza el nombre de los productos de interes en prod_ofrecido
replace_dict = { ".+afc" : "afc",
                  "^aumento.+cupo.+tdc": "aumento_cupo",
                  "^tdc.+|tdc": "tdc",
                  ".+veh\wcu.+" : "vehiculo",
                  "cdt" : "cdt",
                  "multiasis.+" : "mltasistencia",
                  "segur.+" : "seguros",
                  "(^inversi\wn.+)|(^fond.+)|(^renta\s\w.+)" : "inversion" ,
                  "^libre\sinve.+" : "libre_inversion",
                  "^libran.+" : "libranza",
                  "credi\wgil" : "crediagil", 
                  "^redifer.+" : "rediferido",
                  "^retan.+" : "retanqueo"  }

for i, (reg,new_string) in enumerate(replace_dict.items()):

    if i == 0:
        data_bpo["acn_ofr"] = data_bpo.\
                                prod_ofrecido.\
                                    str.\
                                        replace(pat = reg, repl = new_string, regex = True)
        
    else:

        data_bpo["acn_ofr"] = data_bpo.\
                                acn_ofr.\
                                    str.\
                                        replace(pat = reg, repl = new_string, regex = True)

#se normaliza el nombre de los productos de interes en prod_aceptado
for i, (reg,new_string) in enumerate(replace_dict.items()):

    if i == 0:
        data_bpo["acn_acpt"] = data_bpo.\
                                    prod_aceptado.\
                                        str.\
                                            replace(pat = reg, repl = new_string, regex = True)
        
    else:

        data_bpo["acn_acpt"] = data_bpo.\
                                acn_acpt.\
                                    str.\
                                        replace(pat = reg, repl = new_string, regex = True) 


#se filtra donde estan los productos de interes (o acciones) OFRECIDAS
data_bpo = data_bpo[
                    (data_bpo["acn_ofr"] == "afc") |
                    (data_bpo["acn_ofr"] == "aumento_cupo") |
                    (data_bpo["acn_ofr"] == "tdc") |
                    (data_bpo["acn_ofr"] == "vehiculo") |
                    # (data_bpo["acn_ofr"] == "mltasistencia") |
                    (data_bpo["acn_ofr"] == "seguros") |
                    (data_bpo["acn_ofr"] == "inversion") |
                    (data_bpo["acn_ofr"] == "libre_inversion") |
                    (data_bpo["acn_ofr"] == "libranza") |
                    (data_bpo["acn_ofr"] == "crediagil") |
                    (data_bpo["acn_ofr"] == "rediferido")|
                    (data_bpo["acn_ofr"] == "retanqueo") |
                    (data_bpo["acn_ofr"] == "no_acepta_producto")
                    ]

#creación de flag para identificar casos en los que se ofrece un producto y se acepta otros
def ofrece_dif_acepta (x):
    ofrece = x[0]
    acepta = x[1]

    if (acepta != "no_acepta_producto") & (ofrece != acepta):
        flag = 1
    else:
        flag = 0
    return flag
data_bpo["flag_ofr_dif_acpt"] = data_bpo[["acn_ofr","acn_acpt"]].apply(ofrece_dif_acepta, axis = 1)

#sefiltran los datos donde el flag es 0, es decir, registros donde unicamente aceptan el producto ofrecido o no aceptan producto
data_bpo = data_bpo[data_bpo["flag_ofr_dif_acpt"] == 0]
data_bpo.drop("flag_ofr_dif_acpt", inplace = True, axis = 1)



###############################  Creación de recompensa
def reward_function (x):
    prod_ofrecido = x[0]
    prod_aceptado = x[1]
    causal_no_venta = x[2]

    if prod_ofrecido == prod_aceptado: 
        reward = 5 
    elif (prod_ofrecido != prod_aceptado) and (causal_no_venta == "cliente pide que lo vuelvan a llamar más tarde") or (causal_no_venta == "Ccontactar de nuevo para cierre de venta"):
        reward = 1
    else:
        reward = -1

    return reward

data_bpo["reward"] = data_bpo[["acn_ofr","acn_acpt", "causal_no_venta"]].apply(reward_function, axis=1 )
data_bpo.head()


#se eliminan columnas que no interesan
data_bpo.drop(["prod_ofrecido","prod_aceptado", "causal_no_venta", "acn_acpt"], axis = 1, inplace = True)
#se crea columna de accion ("acn") para creacion de dummies
data_bpo["acn"] = data_bpo["acn_ofr"].copy()
data_bpo = pd.get_dummies(data_bpo, columns = ["acn"])

#se agrupa el dataset para tener un registro por accion
data_bpo = data_bpo.groupby(["num_doc","periodo","mes", "dia","acn_ofr"]).sum().reset_index()

#Se reemplazan los valores mayores a 1 por 1 en el vector de acciones
for accion in list(data_bpo.acn_ofr.unique()):
    data_bpo["acn_"+accion] = np.where(data_bpo["acn_"+accion]>=1,1,0)

#se reemplazan los valores de reward mayores a 5 por un reward maximo de 5
data_bpo["reward"] = np.where(data_bpo["reward"]>5, 5,data_bpo["reward"] )
data_bpo.drop("acn_ofr", axis = 1, inplace = True)

#se organizan nombre de columnas
data_bpo.columns = ['id',
                    'periodo',
                    'mes',
                    'dia',
                    'reward_bpo',
                    'acn_aumento_cupo_bpo',
                    'acn_crediagil_bpo',
                    'acn_libranza_bpo',
                    'acn_libre_inversion_bpo',
                    'acn_rediferido_bpo',
                    'acn_retanqueo_bpo',
                    'acn_tdc_bpo']





################################################################################
####################################    STOC

# data_stoc = sp.helper.obtener_dataframe(""" 
# SELECT
# CAST(num_doc as bigint) as id,
# periodo,
# MONTH(CAST(f_gestion as timestamp)) as mes,
# DAY(CAST(f_gestion as timestamp)) as dia,
# descri_estado,
# descri_subestado,
# descri_producto,
# observaciones
# FROM resultados_clientes_personas_y_pymes.fco_gestion_oc_stoc_personas
# WHERE periodo BETWEEN 202200 AND 202212
# """)


# #se eliminan registros donde descri_producto es nulo
# data_stoc.dropna(axis = 0, subset = ["descri_producto"], inplace = True)

# #Se vuelve minuscula todos los campos
# data_stoc["descri_producto"] = data_stoc.descri_producto.str.lower()
# data_stoc["descri_estado"] = data_stoc.descri_estado.str.lower()
# data_stoc["descri_subestado"] = data_stoc.descri_subestado.str.lower()
# data_stoc["observaciones"] = data_stoc.observaciones.str.lower()

# #se filtra a registros donde el estado y subestado sean diferentes a "contacto no exitoso" y a "no contesta", respectivamente
# data_stoc = data_stoc[data_stoc["descri_estado"] != "contacto no exitoso"]
# data_stoc = data_stoc[data_stoc["descri_subestado"] != "no contesta"]
# data_stoc = data_stoc[data_stoc["descri_subestado"] != "número equivocado"]

# #se normaliza el nombre de los productos de interes
# replace_dict = {   ".+afc" : "afc",
#                    "^aumento.+cupo.+|^aumento.+cupo.+" : "aumento_cupo",
#                    "^tarjeta.+cr\wdito": "tdc",
#                    ".+veh\wcu.+" : "vehiculo",
#                    "^cd\w.*" : "cdt",
#                    "^multiasis.+" : "mltasistencia",
#                    "^segur.+" : "seguros",
#                    "(^inversi\wn.+)|(^fond.+)|(^renta\s\w.+)" : "inversion" ,
#                    "^libr\w.+inver.+" : "libre_inversion",
#                    "^libra\w.+" : "libranza",
#                    "^credi\wg\w.*" : "crediagil", 
#                    "^redifer.+" : "rediferido"}

# for i, (reg,new_string) in enumerate(replace_dict.items()):

#     if i == 0:
#         data_stoc["acn"] = data_stoc.\
#                             descri_producto.\
#                              str.\
#                               replace(pat = reg, repl = new_string, regex = True)
        
#     else:

#         data_stoc["acn"] = data_stoc.\
#                             acn.\
#                              str.\
#                               replace(pat = reg, repl = new_string, regex = True)


# #se filtra donde estan los productos de interes (o acciones)
# data_stoc = data_stoc[(data_stoc["acn"] == "afc") |
#                       (data_stoc["acn"] == "aumento_cupo") |
#                       (data_stoc["acn"] == "tdc") |
#                       (data_stoc["acn"] == "vehiculo") |
#                       (data_stoc["acn"] == "mltasistencia") |
#                       (data_stoc["acn"] == "seguros") |
#                       (data_stoc["acn"] == "inversion") |
#                       (data_stoc["acn"] == "libre_inversion") |
#                       (data_stoc["acn"] == "libranza") |
#                       (data_stoc["acn"] == "crediagil") |
#                       (data_stoc["acn"] == "rediferido")
#                       ]

# #se obtienen variables dummies de la columna acn
# # data_analisis = data_stoc.copy()
# data_stoc["productos"] = data_stoc["acn"].copy()
# data_stoc = pd.get_dummies(data_stoc, columns = ["acn"])



# #Creación de recompensa
# def reward_function (x):
#     estado = x[0]
#     subestado = x[1]
#     subestados = ['otro, ¿cuál? (escríbelo en observaciones)', #0
#                     'no satisface sus necesidades', #1
#                     'cierre exitoso', #2
#                     'rechazado en primera instancia', #3
#                     'cliente interesado pero ofrecer después', #4
#                     'pendiente firma de documento/estudio', #5
#                     'cliente no cumple condiciones', #6
#                     'pendiente autorización empresa (libranza especial', #7
#                     'cliente en proceso de apertura con otro canal', #8
#                     'mejor oferta competencia', #9
#                     'pendiente realce tarjetas',
#                     'cliente insatisfecho con el banco', #10
#                     'cola congestionada/fila congestionada', #11
#                     'número equivocado']
#     finalizado_ext_dict = {"max_reward" : [subestados[2]],
#                             "one_reward" : [subestados[4], subestados[5], subestados[8]],
#                             "negative_reward": [subestados[-3],
#                                                 subestados[6],
#                                                 subestados[-2],
#                                                 subestados [1]]
#                          }

#     volver_ofrecer_dict = {"one_reward" : [subestados[4],
#                                            subestados[0],
#                                            subestados [7]],
#                             "negative_reward": [subestados[-3],
#                                                 subestados [1]]     
#                             }
  
    
#     if (estado == "finalizado exitoso") and (subestado in finalizado_ext_dict["max_reward"]):
        
#         reward = 5
#     elif (estado == "finalizado exitoso") and (subestado in finalizado_ext_dict["one_reward"]):
#         reward = 1
#     elif (estado == "finalizado exitoso") and (subestado in finalizado_ext_dict["negative_reward"]):
#         reward = -1
#     elif estado == "finalizado no exitoso":
#         reward = -1
#     elif (estado == "volver a ofrecer") and (subestado in volver_ofrecer_dict["one_reward"] ):
#         reward = 1
#     elif (estado == "volver a ofrecer") and (subestado in volver_ofrecer_dict["negative_reward"] ):
#         reward = -1
#     else:
#         reward = 0
#     return reward

# data_stoc["reward"] = data_stoc[["descri_estado","descri_subestado"]].apply(reward_function, axis=1 )

# #se eliminan columnas que no interesan 
# data_stoc.drop(["descri_estado","descri_subestado", "descri_producto", "observaciones"], axis = 1, inplace = True)

# #se agrupa la data para tener solo un registro por periodo mensual
# data_stoc.drop("productos", axis = 1, inplace = True)
# data_stoc = data_stoc.groupby(["id", "periodo", "mes", "dia"]).sum().reset_index()


# #se reemplaza donde la acn_ es mayor a 1, por 1
# acciones_list = ["afc",
#                   "aumento_cupo",
#                   "tdc",
#                   "vehiculo",
#                   "seguros",
#                   "inversion",
#                   "libre_inversion",
#                   "libranza",
#                   "crediagil",
#                   "rediferido"]
# for accion in acciones_list:
#     data_stoc["acn_"+accion] = np.where(data_stoc["acn_"+accion]>=1, 1, 0)


# #se organizan nombre de columnas
# data_stoc.columns = ['id', 'periodo', 'mes', 'dia', 'acn_afc_stoc', 'acn_aumento_cupo_stoc',
#        'acn_crediagil_stoc', 'acn_inversion_stoc', 'acn_libranza_stoc', 'acn_libre_inversion_stoc',
#        'acn_rediferido_stoc', 'acn_seguros_stoc', 'acn_tdc_stoc', 'acn_vehiculo_stoc', 'reward_stoc']






################################################################################
####################################    CONSOLIDACION

#Se concatenan las bases de datos
# data_acciones_reward = pd.concat([data_bpo, data_stoc], ignore_index= True)

data_acciones_reward = data_bpo.copy()



data_acciones_reward.fillna(0, inplace = True)


# data_acciones_reward["reward"] = data_acciones_reward["reward_bpo"] +  data_acciones_reward["reward_stoc"]
data_acciones_reward["reward"] = data_acciones_reward["reward_bpo"].copy()

# data_acciones_reward.drop(["reward_bpo", "reward_stoc"], axis = 1, inplace = True)

data_acciones_reward.drop(["reward_bpo"], axis = 1, inplace = True)

def periodo_menos_uno(x):

    if x > 202201:
        return x-1
    elif x == 202201:
        return 202112

data_acciones_reward["periodo_menos_1"] = data_acciones_reward["periodo"].apply(periodo_menos_uno)

# acn_list =   ['acn_aumento_cupo_bpo',
#        'acn_crediagil_bpo', 'acn_libranza_bpo', 'acn_libre_inversion_bpo',
#        'acn_rediferido_bpo', 'acn_retanqueo_bpo', 'acn_tdc_bpo',
#        'acn_afc_stoc', 'acn_aumento_cupo_stoc', 'acn_crediagil_stoc',
#        'acn_inversion_stoc', 'acn_libranza_stoc', 'acn_libre_inversion_stoc',
#        'acn_rediferido_stoc', 'acn_seguros_stoc', 'acn_tdc_stoc',
#        'acn_vehiculo_stoc']


acn_list =   [
                'acn_aumento_cupo_bpo',
                'acn_crediagil_bpo',
                'acn_libranza_bpo',
                'acn_libre_inversion_bpo',
                'acn_rediferido_bpo',
                'acn_retanqueo_bpo',
                'acn_tdc_bpo']

data_acciones_reward["num_accion"] = data_acciones_reward[acn_list].idxmax(axis = 1)
data_acciones_reward["num_accion"] = data_acciones_reward["num_accion"].apply(lambda x: acn_list.index(x))

data_acciones_reward = data_acciones_reward.groupby(["id", "periodo", "periodo_menos_1", "mes", "num_accion"]).sum().reset_index()



#Se sube el dataframe de acciones y recompensa al area de proceso_clientes_personas_y_pymes
sp.subir_df(data_acciones_reward,nombre_tabla="dabj_test_acc_rwrd_int", zona = "proceso_clientes_personas_y_pymes", modo = 'overwrite')



