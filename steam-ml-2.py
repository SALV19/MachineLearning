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
print(df_games.head())

# %% [markdown]
# ## Función auxiliar para detectar la calidad del dataset y columnas con una cantidad considerable de datos faltantes

# %%
def see_nan_values(df):
	# df.info()
	nan_values = df.isnull().sum()
	# print(nan_values / len(df) * 100)

	# print(f"Lenght of dataset: {len(df)}")

# %%
see_nan_values(df_games)

# %% [markdown]
# Ya que la cantidad de valores vaciós en precio es considerable y un dato que quiero analizar decidí tumbar las instancias que no contaban con este valor.

# %%
# Test remove empty price values
df_games.dropna(subset=["price_overview"], inplace=True)

# %%
nan_values = df_games.isnull().sum()
# print(nan_values / len(df_games) * 100)

# print(f"Lenght of dataset: {len(df_games)}")

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
see_nan_values(df_games)

# %% [markdown]
# ## Análisis genérica de la información

# %%
df_games['currency'].hist()
unique_curr = df_games['currency'].value_counts()
print(unique_curr)


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

# %%
df_games.info()

# %%
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
see_nan_values(df_games)

# %% [markdown]
# # DF Games está limpio y listo, leer los siguientes archivos y unión de la información

# %%
import seaborn as sns

print(len(df_games))
df_games.info()
df_games.drop(columns=["price", "currency", 'is_free'])

print(df_games['prices_eur'].head())
print("Cantidad de juegos con la categoría gratis", df_games['is_free'].value_counts())
print('Min', df_games['prices_eur'].min())
print('Max', df_games['prices_eur'].max())
print('Quantil 0.99: ', df_games["prices_eur"].quantile(0.99))

df_games[df_games["prices_eur"] < df_games["prices_eur"].quantile(0.99)]['prices_eur'].hist()


# %%
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

# %%
print("genres:")
see_nan_values(df_genres) # Clean
df_genres["genre"].value_counts()

# %%
print("reviews:")
df_reviews.dropna(thresh=1)
see_nan_values(df_reviews)
df_reviews_important = df_reviews.drop(['metacritic_score', 'reviews', 'recommendations', 'steamspy_user_score', 'steamspy_score_rank', 'steamspy_positive', 'steamspy_negative'], axis=1)
print(df_reviews_important.columns)
df_reviews_important

# %%
df_reviews_important.info()
see_nan_values(df_reviews_important)
print(df_reviews_important['review_score_description'].value_counts())
categories = df_reviews_important['review_score_description'].unique()
print("Categorías: ", categories)
df_reviews_important.dropna(subset=['review_score_description'], inplace=True)
categories = df_reviews_important['review_score_description'].unique()
print(categories)

# %%
print("insights:")
see_nan_values(df_insights)
print('playtime_average_forever: ', df_insights['playtime_average_forever'].value_counts())
print('playtime_average_2weeks: ', df_insights['playtime_average_2weeks'].value_counts())
print('playtime_median_forever: ', df_insights['playtime_median_forever'].value_counts())
print('playtime_median_2weeks: ', df_insights['playtime_median_2weeks'].value_counts())

# Tumbar columnas que no aportan información relevante
df_insights_clean = df_insights.drop(['developer', 'publisher', 'price', 'initial_price', 'discount', 'languages', 'genres', 'playtime_average_forever', 'playtime_average_2weeks', 'playtime_median_forever', 'playtime_median_2weeks'], axis=1)
df_insights_clean

# %%
print("tags:")
see_nan_values(df_tags)
df_tags_count = df_tags["tag"].value_counts()
print("Cantidad de tags repetidos: ", len(df_tags_count[df_tags_count > 1]))

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
plt.figure(figsize=(14, 6))
sns.scatterplot(x=games['positive'], y=games['negative'], hue=games['owners_range'], style=games['review_score_description'])

plt.legend(
	bbox_to_anchor=(1.05, 1),
	loc='upper left'
)

plt.tight_layout()
plt.show()

# %%
games_one_hot = one_hot_encoding(games, 'owners_range')
games_one_hot = one_hot_encoding(games_one_hot, 'review_score_description')
print(games_one_hot.columns[3:])
plt.figure(figsize=(18, 9))
games_one_hot[games_one_hot.columns[3:]]
correlation = games_one_hot[games_one_hot.columns[3:]].corr()
sns.heatmap(correlation)

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

# 

df_x = games_final.iloc[:, [0, *range(2, len(games_final.columns))]]
df_y = games_final.iloc[:, 1]
# 

train_length = round(len(df_x) * .6)
validation_length = round(train_length / 2) + train_length
df_x_train = df_x[:train_length]
df_x_validation = df_x[train_length:validation_length]
df_x_test = df_x[validation_length:]


df_y_train = df_y[:train_length]
df_y_validation = df_y[train_length:validation_length]
df_y_test = df_y[validation_length:]

df_x_standar, mean, std = standardize_data(df_x_train.iloc[:, :5])
df_x_train = pd.concat([df_x_standar.iloc[:, :], df_x_train.iloc[:, 5:]], axis=1)

df_x_validation_standar = (df_x_validation.iloc[:, :5] - mean) / std
df_x_validation = pd.concat([df_x_validation_standar, df_x_validation.iloc[:, 5:]], axis=1)

df_x_test_standar = (df_x_test.iloc[:, :5] - mean) / std
df_x_test = pd.concat([df_x_test_standar, df_x_test.iloc[:, 5:]], axis=1)

see_nan_values(df_x_train)

# %%
# GET PCA to see tendencies 
covariance_matrix = np.cov(df_x.T.astype(float))

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Calculating the explained variance on each of components
variance_explained = eigen_values / eigen_values.sum() * 100

# Identifying components that explain the relationship between the data

cumulative_variance_explained = np.cumsum(variance_explained)

projection_matrix = (eigen_vectors.T).T
projection_matrix

# Total -> 85.8% (PC_x = 78.07 + 7.78)
df_pca = df_x.dot(projection_matrix)

# %%
components = pd.DataFrame(df_pca)
components.head()

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

# %% [markdown]
# $$ \sum_{i=1}^n \left( h(x_i) - y_i  \right) x_{ij} $$

# %%
def sum(params, x_values, b, y):
	"""sum

	Args:
		params (array<float>): m_s
		x_values (matrix<float>):  len(params) * len(instances)
		b (float)

	Returns:
		float: error total
	"""
	# Calcular error interno
	errors = h(params, x_values, b) - y

	mean_error = calculate_loss(errors)

	# Calcular error acumulado
	theta_error = np.zeros(len(params) + 1)
	theta_error[:-1] = np.dot(errors, x_values)
	
	# Suma parámetro b
	theta_error[-1] = errors.sum()

	return theta_error, mean_error

# %% [markdown]
# $$ \theta_j = \theta_j - \frac{\alpha}{m} \sum_{i=1}^n \left( h(x_i) - y_i  \right) x_{ij} $$

# %%
def GD(params, alfa, b, x_values, y):
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

	error_evaluation, mean_error = sum(params, x_values, b, y)
	new_params = params - alfa * m * error_evaluation[:-1]
	b = b - alfa * m * error_evaluation[-1]

	return new_params, b, mean_error

# %%
def train(df_x, df_y, df_x_validation, df_y_validation, df_x_test, df_y_test):
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
	errors = []
	test_errors = []
	validation_errors = []
	
	while True:
		oldparams = params.copy()
		# print("Params: ", params)
  
		params, b, mean_error = GD(params, alfa, b, df_x, df_y)	
		errors.append(mean_error)

		# plt.plot(errors)
		validation_predictions = h(params, df_x_validation, b)
		validation_loss = calculate_loss(validation_predictions - df_y_validation)
		validation_errors.append(validation_loss)
  
		test_predictions = h(params, df_x_test, b)
		test_loss = calculate_loss(test_predictions - df_y_test)
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
  
		if np.allclose(oldparams, params, atol=1e-3):
			print("Reached minimum")
			print("Epoch: ", epochs)
			print ("final params:")
			print (params)
			break
	return params, b, errors, test_errors, validation_errors

# %%
params, b, errors, test_errors, validation_errors = train(df_x_train, df_y_train, df_x_validation, df_y_validation, df_x_test, df_y_test)

plt.figure(figsize=(10, 6))

plt.plot(errors, label="Train Loss")
plt.plot(validation_errors, label="Validation Loss")
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
plt.plot(validation_errors, label="Validation Loss")
plt.plot(test_errors, label="Test Loss")

plt.yscale("log")
plt.ylim(1e-1, 1e5)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs Validation vsTest Loss")

plt.legend()
plt.grid()

plt.show()

# %%
train_predictions = h(params, df_x_train, b)
validation_predictions = h(params, df_x_validation, b)
test_predictions = h(params, df_x_test, b)

print("Train loss MSE:", calculate_loss(train_predictions - df_y_train))
print("Validation loss MSE:", calculate_loss(validation_predictions - df_y_validation))
print("Test loss MSE:", calculate_loss(test_predictions - df_y_test))

# %%
print(f"Tamaño validation x: {len(df_x_validation)}")
print(f"Tamaño validation y: {len(df_y_validation)}")
results = pd.DataFrame(data = np.dot(df_x_validation, params) + b, columns=["Resultado"])
results["Esperado"] = df_y_validation.reset_index(drop=True)
results["Error MSE"] = (results["Resultado"] - results["Esperado"]) ** 2


print(f"Max error MSE: {results["Error MSE"].max()}")
print(f"Mean error MSE: {results["Error MSE"].mean()}")
print(f"Mediana error MSE: {results["Error MSE"].median()}")
print(f"Min error MSE: {results["Error MSE"].min()}")

# %%
print(f"Tamaño test x: {len(df_x_test)}")
print(f"Tamaño test y: {len(df_y_test)}")
results = pd.DataFrame(data = np.dot(df_x_test, params) + b, columns=["Resultado"])
results["Esperado"] = df_y_test.reset_index(drop=True)
results["Error MSE"] = (results["Resultado"] - results["Esperado"]) ** 2


print(f"Max error MSE: {results["Error MSE"].max()}")
print(f"Mean error MSE: {results["Error MSE"].mean()}")
print(f"Mediana error MSE: {results["Error MSE"].median()}")
print(f"Min error MSE: {results["Error MSE"].min()}")

# %%
plt.figure(figsize=(9, 6))

# Dibujar el Scatter Plot de los datos reales vs predichos
plt.scatter(
    results["Esperado"],
    results["Resultado"],
    color="#2b5c8f",
    alpha=0.5,
    edgecolors="none",
    label="Puntos de Test",
)

# Dibujar la línea ideal de predicción perfecta (y = x)
lim_min = min(results["Esperado"].min(), results["Resultado"].min()) - 1
lim_max = max(results["Esperado"].max(), results["Resultado"].max()) + 1
plt.plot(
    [-1, 11],
    [lim_min, lim_max],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Predicción Perfecta ($y = x$)",
)

# Formato de la gráfica
plt.title(
    "Diagnóstico del Modelo: Valores Reales vs. Predicciones", fontsize=13
)
plt.xlabel("Valor Real (Esperado)", fontsize=11)
plt.ylabel("Valor Predicho ($\hat{y}$)", fontsize=11)
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

plt.show()

# %%
import os
import json

with open("params_2.json", "w") as f:
  json.dump(params.tolist(), f)

# %%
import json

with open("params-original.json", 'r') as file:
  data = json.load(file)
  b = data["b"]
  params = data["params"]
  
  print(f"Tamaño test x: {len(df_x_test)}")
  print(f"Tamaño test y: {len(df_y_test)}")
  results = pd.DataFrame(data = np.dot(df_x_test, params) + b, columns=["Resultado"])
  results["Esperado"] = df_y_test.reset_index(drop=True)
  results["Error MSE"] = (results["Resultado"] - results["Esperado"]) ** 2
  
  
  diferencia = results["Resultado"] - results["Esperado"]
  
  n = len(diferencia)

  sd = np.sqrt(np.sum((diferencia - diferencia.mean()) ** 2) / (n - 1))

  se = sd / np.sqrt(n)

  t = diferencia.mean() / se

  print(f"Max error MSE: {results["Error MSE"].max()}")
  print(f"Mean error MSE: {results["Error MSE"].mean()}")
  print(f"Mediana error MSE: {results["Error MSE"].median()}")
  print(f"Min error MSE: {results["Error MSE"].min()}")
  
  print(f"\nDesviación estandar: {sd}")
  print(f"T-Student: {t}")


