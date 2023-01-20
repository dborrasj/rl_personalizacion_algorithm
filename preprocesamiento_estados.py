import pandas as pd
import numpy as np

from sparky_bc import Sparky
import re



sp = Sparky('dborras', 'IMPALA_PROD')


################################################################################
####################################    INTENCIONES ACT

data_intenciones_act = sp.helper.obtener_dataframe("""
SELECT *
FROM proceso_clientes_personas_y_pymes.dabj_rl_intenciones_act
""")

data_intenciones_past = sp.helper.obtener_dataframe("""
SELECT *
FROM proceso_clientes_personas_y_pymes.dabj_rl_intenciones_past
""")


data_intenciones_concat = pd.concat(data_intenciones_past, data_intenciones_act)
#se eliminan filas con nan
data_intenciones_act.dropna(inplace=True)

#se pone en minuscula todos los campos
data_intenciones_act["descri_tipo_intencion"] = data_intenciones_act.descri_tipo_intencion.str.lower()
data_intenciones_act["descri_producto_detallado"] = data_intenciones_act.descri_producto_detallado.str.lower()

#se normaliza el nombre de los productos de interes
replace_dict = { "^aumento.+cu.+" : "aumento_cupo",
                   "^cd\w.*" : "cdt",
                   "^credi\wg\w.+" : "crediagil",
                   ".+veh\wcu.+" : "vehiculo",
                   ".+afc" : "afc",
                   "^segur.+" : "seguros",
                   "(^inversi\wn.+)|(^renta\s\w.+)" : "inversion" ,
                   "^tarjeta.+cr\wdito": "tdc",
                   "^libre\s.+" : "libre_inversion",
                   "^libra\w.+" : "libranza" }
for i, (reg,new_string) in enumerate(replace_dict.items()):
    
    if i == 0:

        data_intenciones_act["int"] = data_intenciones_act.descri_producto_detallado.str.replace(pat = reg, repl = new_string, regex = True)
    else:

        data_intenciones_act["int"] = data_intenciones_act.int.str.replace(pat = reg, repl = new_string, regex = True)


#Se dejan las intenciones de los productos de interes
data_intenciones_act = data_intenciones_act[(data_intenciones_act["int"] == "afc") |
                    (data_intenciones_act["int"] == "aumento_cupo") |
                    (data_intenciones_act["int"] == "tdc") |
                    (data_intenciones_act["int"] == "vehiculo") |
                    # (data_intenciones_act["int"] == "mltasistencia") |
                    (data_intenciones_act["int"] == "seguros") |
                    (data_intenciones_act["int"] == "inversion") |
                    (data_intenciones_act["int"] == "libre_inversion") |
                    (data_intenciones_act["int"] == "libranza") |
                    (data_intenciones_act["int"] == "crediagil") |
                    (data_intenciones_act["int"] == "rediferido")
                    ]


#se toman datos unicamente de nuevo producto y que la intención quedó abierta (flag_no_fin)
data_intenciones_act = data_intenciones_act[data_intenciones_act["descri_tipo_intencion"] == "nuevo negocio/producto"]

#Se eliminan columnas que no son de interes
data_intenciones_act.drop(["descri_tipo_intencion", "id_intencion"], axis = 1, inplace = True)

#se obtienen variables dummies de la columna int
data_intenciones_act = pd.get_dummies(data_intenciones_act, columns = ["int"] )

data_intenciones_act.drop(["descri_producto_detallado"], axis = 1, inplace = True)

#se agrupa por id, periodo y mes 
data_intenciones_act = data_intenciones_act.groupby(["id","periodo", "mes_intencion"]).sum().reset_index()



################################################################################
####################################    INTENCIONES ACT
data_intenciones_act = sp.helper.obtener_dataframe("""
SELECT *
FROM proceso_clientes_personas_y_pymes.dabj_rl_intenciones_past
""")







################################################################
############################ CONSOLIDADO




data_state_menos_1 = data_state_past[data_state_past["periodo"] < 202211]
data_state_act = data_state_past[data_state_past.id.isin(data_state_past.id)]
data_state_menos_1 = data_state_past[data_state_past.id.isin(data_state_act.id)]

#Se eliminan registros en donde NO haya el par estado actual y anterior (menos_1)
def periodo_menos_uno(x):

    if x > 202201:
        return x-1
    elif x == 202201:
        return 202112

data_state_act["periodo_menos_1"] = data_state_act["periodo"].apply(periodo_menos_uno)
data_verificacion = data_state_act[["id", "periodo_menos_1"]]
data_state_menos_1 = pd.merge(data_state_menos_1, 
                                data_verificacion,
                                how = "inner",
                                left_on = ["id", "periodo"],
                                right_on = ["id", "periodo_menos_1"])

data_state_menos_1.drop("periodo_menos_1", axis = 1, inplace = True)



