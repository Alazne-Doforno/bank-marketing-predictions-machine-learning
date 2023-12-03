# MODELO 2 - Definiendo las decisiones sobre a quién dirigir el primer contacto.

import pickle
import pandas as pd
import os
import datetime
dia_de_hoy = datetime.datetime.now().date()

df_2 = pd.read_csv('/Users/alazne/Data Science/Bootcamp-Alazne/Proyectos/Entregas/Machine-learning/notebooks/x_rus_train_modelo2.csv')

# Os busca la ruta donde está alojado el archivo donde estoy ejecutando
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, 'modelo_2.pkl')
model = pickle.load(open(model_path, 'rb'))

predictions = model.predict(df_2)
pd.DataFrame(predictions, columns=['Predicciones']).to_csv(f'Proyectos/Entregas/Machine-learning/data/predicciones_llamada{dia_de_hoy}.csv', index=False)

