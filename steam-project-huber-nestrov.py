# %%
import re
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import math

# %% [markdown]
# ## Limpieza excel para no afectar formato devido a objetos json

# %%
def clean_parse_file():
	input_file = "./steam-dataset/games.csv"
	output_file = "./steam-dataset/games_fixed.csv"

	# GPT me dió el regex para limpiar el excel y poder leerlo más facilmente con 
	# pandas
	pattern = re.compile(r',(?=[^{}]*\})') 

	with open(input_file, "r", encoding="utf-8") as infile, \
		open(output_file, "w", encoding="utf-8") as outfile:

		for line in infile:
			line = pattern.sub("|", line)
			outfile.write(line)

# %% [markdown]
# ## Leer archivos .csv en objeto pandas

# %%
df_games = pd.read_csv("./steam-dataset/games_fixed_clean.csv", encoding="ISO-8859-1", na_values="\\N")

# %% [markdown]
# ## Función auxiliar para detectar la calidad del dataset y columnas con una cantidad considerable de datos faltantes

# %%
def see_nan_values(df):
	nan_values = df.isnull().sum()

# %% [markdown]
# Ya que la cantidad de valores vaciós en precio es considerable y un dato que quiero analizar decidí tumbar las instancias que no contaban con este valor.

# %%
# Test remove empty price values
df_games.dropna(subset=["price_overview"], inplace=True)

# %%
nan_values = df_games.isnull().sum()

# %% [markdown]
# Debido a que el precio viene en una columna en formato JSON tuve que utilizar regex para extraer el precio y moneda de las instancias para poder analizarlo correctamente. De esta manera extraje el precio en dos columnas con las que puedo trabajar y analizar.

# %%
# IA para extraer la información con regex
s = df_games["price_overview"].astype("string")

df_games["price"] = pd.to_numeric(
	s.str.extract(r'final\\?"?\s*[:|]\s*(\d+(?:\.\d+)?)', expand=False)
) / 100

df_games["currency"] = s.str.extract(
	r'currency\\?"?\s*[:|]\s*\\?"?([A-Z]{3})',
	expand=False
)

df_games

# %%
df_games.drop(columns=["price_overview", "languages", "type"], inplace=True)

# %% [markdown]
# ## Análisis genérica de la información

# %%
unique_curr = df_games['currency'].value_counts()


# %% [markdown]
# ## Hay juegos con monedas diferentes
# 
# Por ello, decidí normalizar los precios de los juegos a Euros, ya que es la moneda más común en el dataset. 
# 
# Archivo descargado el 22 de agosto, 2026
# https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip 

# %%
from currency_converter import CurrencyConverter
c = CurrencyConverter('./eurofxref-hist.csv')


# Missing rates
df_games = df_games.replace('SAR', 'ZAR')

rates = {
	currency: c.convert(1, currency, 'EUR')
	for currency in c.currencies
}

rates['PEN'] = 0.26
rates['UAH'] = 0.019
rates['COP'] = 0.00028
rates['KWD'] = 2.78
rates['KZT'] = 0.0019
rates['TWD'] = 0.027
rates['AED'] = 0.23
rates['VND'] = 0.000033
df_games['prices_eur'] = df_games['price'] * df_games['currency'].map(rates)

import seaborn as sns

df_games.drop(columns=["price", "currency", 'is_free'])

df_games[df_games["prices_eur"] < df_games["prices_eur"].quantile(0.99)]['prices_eur'].hist()



def one_hot_encoding(df, column): 
	# https://www.geeksforgeeks.org/machine-learning/one-hot-encoding-from-a-pandas-column-containing-a-list/
	for category in df[column].unique():
		df[category] = df[column].apply(lambda x: 1 if category in x else 0)

	# Drop the original column
	return df.drop(column, axis=1)

# %%
df_categories = pd.read_csv('./steam-dataset/categories.csv', na_values="\\N")
df_genres = pd.read_csv("./steam-dataset/genres.csv", na_values="\\N")
df_reviews = pd.read_csv("./steam-dataset/reviews.csv", na_values="\\N")
df_insights = pd.read_csv("./steam-dataset/steamspy_insights.csv", encoding="ISO-8859-1", na_values="\\N")
df_tags = pd.read_csv("./steam-dataset/tags.csv", na_values="\\N")

df_genres["genre"].value_counts()

df_reviews.dropna(thresh=1)

df_reviews_important = df_reviews.drop(['metacritic_score', 'reviews', 'recommendations', 'steamspy_user_score', 'steamspy_score_rank', 'steamspy_positive', 'steamspy_negative'], axis=1)
df_reviews_important

# %%
df_reviews_important.info()

categories = df_reviews_important['review_score_description'].unique()

df_reviews_important.dropna(subset=['review_score_description'], inplace=True)
categories = df_reviews_important['review_score_description'].unique()

# %%

# Tumbar columnas que no aportan información relevante
df_insights_clean = df_insights.drop(['developer', 'publisher', 'price', 'initial_price', 'discount', 'languages', 'genres', 'playtime_average_forever', 'playtime_average_2weeks', 'playtime_median_forever', 'playtime_median_2weeks'], axis=1)
df_insights_clean

# %%
df_tags_count = df_tags["tag"].value_counts()

# %%
df_games = df_games.drop(["price", "currency", "is_free"], axis=1)

# %%
games = df_games.merge(df_reviews_important.set_index("app_id"), on="app_id", how="inner")
games = games.merge(df_insights_clean.set_index("app_id"), on="app_id", how="inner")
# games = df_games.merge(df_genres.set_index("app_id"), on="app_id", how="inner")
# games = games.merge(df_categories.set_index("app_id"), on="app_id", how="inner")
# games = games.merge(df_tags.set_index("app_id"), on="app_id", how="inner")
games


# %%
games_one_hot = one_hot_encoding(games, 'owners_range')
games_one_hot = one_hot_encoding(games_one_hot, 'review_score_description')
games_one_hot[games_one_hot.columns[3:]]
correlation = games_one_hot[games_one_hot.columns[3:]].corr()


# %%
def standardize_data(df):

	'''
	This function standardize an array, its substracts mean value,
	and then divide the standard deviation.

	param 1: array
	return: standardized array
	'''
	# Fórmula: (x - X_mean) / std
	mean = df.mean()
	std = df.std()
	new_df = (df - mean) / std

	return new_df, mean, std

# %%
# games_one_hot.info()
games_final = games_one_hot.drop(columns=['app_id', 'name', 'release_date'])

games_final = games_final.sample(frac=1)


df_x = games_final.iloc[:, [0, *range(2, len(games_final.columns))]]
df_y = games_final.iloc[:, 1]

train_length = round(len(df_x) * .8)
df_x_train = df_x[:train_length]
df_x_test = df_x[train_length:]

df_y_train = df_y[:train_length]
df_y_test = df_y[train_length:]

df_x_standar, mean, std = standardize_data(df_x_train.iloc[:, :5])
df_x_train = pd.concat([df_x_standar.iloc[:, :], df_x_train.iloc[:, 5:]], axis=1)
df_x_train

df_x_test_standar = (df_x_test.iloc[:, :5] - mean) / std
df_x_test = pd.concat([df_x_test_standar, df_x_test.iloc[:, 5:]], axis=1)


# %%
# GET PCA to see tendencies 
covariance_matrix = np.cov(df_x.T.astype(float))

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Calculating the explained variance on each of components
variance_explained = eigen_values / eigen_values.sum() * 100

projection_matrix = (eigen_vectors.T).T
projection_matrix

# Total -> 85.8% (PC_x = 78.07 + 7.78)
df_pca = df_x.dot(projection_matrix)

# %%
components = pd.DataFrame(df_pca)

sns.scatterplot(x = components[0], y = components[1])

# %% [markdown]
# # Creación del modelo
# 
# Fórmula
# 
# $$ h(x) = mx+b $$

# %%
def h(params, x_values, b):
	""" 
		Calcula el valor de h(x)
		Args:  
			params (array<float>): m_s
			x_values (array<float>): x_ij
			b (float): bias
		
		Returns:
			array<float>: función hipótesis
	"""
	return np.dot(x_values, params) + b

# %%
def calculate_loss(errors):
	return (errors ** 2).mean()

# %%
def huber_loss(y_pred, y, delta = 1.5):
  error = y - y_pred
  abs_error = np.abs(error)
    
    # Condición elemento a elemento:
    # Si abs(error) <= delta usa la pérdida cuadrática, si no, usa la lineal.
  return np.where(
		abs_error <= delta,
		0.5 * (error ** 2),
		delta * (abs_error - 0.5 * delta)
	).mean()

# %% [markdown]
# $$ \sum_{i=1}^n \left( h(x_i) - y_i  \right) x_{ij} $$

# %%
def sum(params, x_values, b, y, delta=1.5):
	"""sum

	Args:
		params (array<float>): m_s
		x_values (matrix<float>):  len(params) * len(instances)
		b (float)

	Returns:
		float: error total
	"""
	# Calcular error interno
	y_pred = h(params, x_values, b)
	errors = y_pred - y

	mean_error = huber_loss(y_pred, y)

	huber_gradient = np.where(
		np.abs(errors) <= delta,
		errors,
		delta * np.sign(errors)
	)

	# Calcular error acumulado
	theta_error = np.zeros(len(params) + 1)
	theta_error[:-1] = np.dot(huber_gradient, x_values)
	
	# Suma parámetro b
	theta_error[-1] = huber_gradient.sum()

	return theta_error, mean_error

# %% [markdown]
# $$ \theta_j = \theta_j - \frac{\alpha}{m} \sum_{i=1}^n \left( h(x_i) - y_i  \right) x_{ij} $$

# %%
def GD(params, alfa, b, x_values, y, change, momentum = 0.9):
	"""GD

	Args:
			params (_type_): _description_
			alfa (_type_): _description_
			b (_type_): _description_
			x_values (_type_): _description_
			y (_type_): _description_

	Returns:
			_type_: _description_
	"""
	# theta_length = len(params);
	m = 1 / len(x_values)
	
	future_params = params + momentum * change[:-1]
	future_b = b + momentum * change[-1]

	error_evaluation, mean_error = sum(future_params, x_values, future_b, y)
	
	change = (momentum * change) - alfa * m * error_evaluation
	new_params = params + change[:-1]
	b = b + change[-1]

	return new_params, b, change, mean_error

# %%
def train(df_x, df_y, df_x_test, df_y_test):
	"""
		Entrenamiento del modelo
		Selecciona instancia
		Manda a llamar GD
	"""
	
	epochs = 0
	max_epochs = 3_000
	alfa = 0.4
	params = np.random.rand(len(df_x.columns)) * 100
	b = 0.5
	
	change = np.zeros(len(params)+1)
	momentum = 0.9
	
	errors = []
	test_errors = []
	
	while True:
		oldparams = params.copy()
		# print("Params: ", params)
  
		params, b, change, mean_error = GD(params, alfa, b, df_x, df_y, change, momentum)	
		errors.append(mean_error)

		# plt.plot(errors)
		test_predictions = h(params, df_x_test, b)
		# test_loss = calculate_loss(test_predictions - df_y_test)
		test_loss = huber_loss(test_predictions, df_y_test)
		test_errors.append(test_loss)
	
		if epochs % 100 == 0:
			print(f"Epoch {epochs}")
			print("Error: ", mean_error)
     
  
		epochs = epochs + 1	
		if epochs == max_epochs:
			print("Final epoch reached")
			print("final params: ")
			print(params)
			break
  
		if np.allclose(oldparams, params, atol=1e-6):
			print("Reached minimum")
			print("Epoch: ", epochs)
			print ("final params:")
			print (params)
			break
	return params, b, errors, test_errors

# %%
params, b, errors, test_errors = train(df_x_train, df_y_train, df_x_test, df_y_test)

plt.figure(figsize=(10, 6))

plt.plot(errors, label="Train Loss")
plt.plot(test_errors, label="Test Loss")

plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs Test Loss")

plt.legend()
plt.grid()

plt.show()

# %%
plt.figure(figsize=(10, 6))

plt.plot(errors, label="Train Loss")
plt.plot(test_errors, label="Test Loss")

plt.yscale("log")
plt.ylim(1e-2, 1e5)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs Test Loss")

plt.legend()
plt.grid()

plt.show()

# %%
df_x_train.columns

# %%
print(params, b)

train_predictions = h(params, df_x_train, b)
test_predictions = h(params, df_x_test, b)

print("Train loss MSE:", calculate_loss(train_predictions - df_y_train))
print("Test loss MSE:", calculate_loss(test_predictions - df_y_test))
print("\nTrain loss Huber:", huber_loss(train_predictions, df_y_train))
print("Test loss Huber:", huber_loss(test_predictions, df_y_test))

# %%
print(f"Tamaño test x: {len(df_x_test)}")
print(f"Tamaño test y: {len(df_y_test)}")

delta = 1.5

results = pd.DataFrame(data = np.dot(df_x_test, params) + b, columns=["Resultado"])
results["Esperado"] = df_y_test.reset_index(drop=True)

error_raw = results["Resultado"] - results["Esperado"]
results["Error MSE"] = error_raw**2
results["Error huber"] = np.where(
    error_raw.abs() <= delta,
    0.5 * (error_raw**2),
    delta * error_raw.abs() - 0.5 * delta**2,
)

diferencia = results["Resultado"] - results["Esperado"]
n = len(diferencia)

sd = np.sqrt(np.sum((diferencia - diferencia.mean()) ** 2) / (n - 1))
se = sd / np.sqrt(n)
t = diferencia.mean() / se

print(f"Max error MSE: {results["Error MSE"].max()}")
print(f"Mean error MSE: {results["Error MSE"].mean()}")
print(f"Mediana error MSE: {results["Error MSE"].median()}")
print(f"Min error MSE: {results["Error MSE"].min()}")
print(f"\nMax error Huber: {results["Error huber"].max()}")
print(f"Mean error Huber: {results["Error huber"].mean()}")
print(f"Mediana error Huber: {results["Error huber"].median()}")
print(f"Min error Huber: {results["Error huber"].min()}")

print(f"\nDesviación estandar: {sd}")
print(f"T-Student: {t}")

# %%
import os
import json

data_to_save = {
    "params": params.tolist(),  # Convierte el np.array a lista
    "b": float(
        b
    ),  # Asegura que sea un float estándar (por si b es un escalar de numpy)
}

with open("params_optimized.json", "w") as f:
  json.dump(data_to_save, f)

def capturar_datos_juego():
    print("=== CAPTURA DE DATOS DEL JUEGO DE STEAM ===")

    price_eur = float(input("Precio en EUR (ej. 19.99): "))
    positive = int(input("Cantidad de reseñas positivas (ej. 15000): "))
    negative = int(input("Cantidad de reseñas negativas (ej. 1200): "))
    total = positive + negative  # Cálculo directo del total
    concurrent_users = int(input("Usuarios concurrentes ayer (ej. 4500): "))

    rangos_owners = [
        "0 .. 20,000",
        "20,000 .. 50,000",
        "50,000 .. 100,000",
        "100,000 .. 200,000",
        "200,000 .. 500,000",
        "500,000 .. 1,000,000",
        "1,000,000 .. 2,000,000",
        "2,000,000 .. 5,000,000",
        "5,000,000 .. 10,000,000",
        "10,000,000 .. 20,000,000",
        "20,000,000 .. 50,000,000",
        "50,000,000 .. 100,000,000",
    ]

    # Corregida la coma faltante en 'Very Negative'
    grade = [
        "Overwhelmingly Positive",
        "Very Positive",
        "Mostly Positive",
        "Positive",
        "Mixed",
        "Negative",
        "Mostly Negative",
        "Very Negative",
        "Overwhelmingly Negative",
        "No user reviews",
    ]

    print("\n--- Selecciona el rango de propietarios (Owners Range) ---")
    for i, rango in enumerate(rangos_owners, 1):
        print(f"[{i}] {rango}")

    while True:
        try:
            opcion = int(input(f"Elige una opción (1-{len(rangos_owners)}): "))
            if 1 <= opcion <= len(rangos_owners):
                owners_selected_idx = opcion - 1
                break
            else:
                print("Opción fuera de rango. Intenta de nuevo.")
        except ValueError:
            print("Por favor, ingresa solo un número entero.")

    print("\n--- Selecciona la calificación de acuerdo a reseñas ---")
    for i, rango in enumerate(grade, 1):
        print(f"[{i}] {rango}")

    while True:
        try:
            opcion = int(input(f"Elige una opción (1-{len(grade)}): "))
            if 1 <= opcion <= len(grade):
                grade_selected_idx = opcion - 1
                break
            else:
                print("Opción fuera de rango. Intenta de nuevo.")
        except ValueError:
            print("Por favor, ingresa solo un número entero.")

    datos_capturados = {
        "price_eur": price_eur,
        "positive": positive,
        "negative": negative,
        "total": total,
        "concurrent_users_yesterday": concurrent_users,
    }

    for idx, rango in enumerate(rangos_owners):
        column_name = rango
        datos_capturados[column_name] = 1 if idx == owners_selected_idx else 0

    for idx, g in enumerate(grade):
        column_name = g
        datos_capturados[column_name] = 1 if idx == grade_selected_idx else 0

    print("\n¡Datos capturados y procesados exitosamente!")
    return pd.DataFrame([datos_capturados])

with open("params-original.json", 'r') as file:
  params = json.load(file)
  entradas = capturar_datos_juego()
  prediction = h(params["params"], entradas[:-1], params["b"])
  
  print("Valor predecido: ", prediction)

