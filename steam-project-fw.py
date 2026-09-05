# %%
import re
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# %%
def clean_parse_file():
	input_file = "./steam-dataset/games.csv"
	output_file = "./steam-dataset/games_fixed.csv"



	pattern = re.compile(r',(?=[^{}]*\})') 

	with open(input_file, "r", encoding="utf-8") as infile, \
		open(output_file, "w", encoding="utf-8") as outfile:

		for line in infile:
			line = pattern.sub("|", line)
			outfile.write(line)


df_games = pd.read_csv("./steam-dataset/games_fixed_clean.csv", encoding="ISO-8859-1", na_values="\\N")


def see_nan_values(df):
	nan_values = df.isnull().sum()


df_games.dropna(subset=["price_overview"], inplace=True)

nan_values = df_games.isnull().sum()


s = df_games["price_overview"].astype("string")

df_games["price"] = pd.to_numeric(
	s.str.extract(r'final\\?"?\s*[:|]\s*(\d+(?:\.\d+)?)', expand=False)
) / 100

df_games["currency"] = s.str.extract(
	r'currency\\?"?\s*[:|]\s*\\?"?([A-Z]{3})',
	expand=False
)

df_games

df_games.drop(columns=["price_overview", "languages", "type"], inplace=True)


unique_curr = df_games['currency'].value_counts()



from currency_converter import CurrencyConverter
c = CurrencyConverter('./eurofxref-hist.csv')

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


df_games.drop(columns=["price", "currency", 'is_free'])

# df_games[df_games["prices_eur"] < df_games["prices_eur"].quantile(0.99)]['prices_eur'].hist()


def one_hot_encoding(df, column): 

	for category in df[column].unique():
		df[category] = df[column].apply(lambda x: 1 if category in x else 0)


	return df.drop(column, axis=1)

df_categories = pd.read_csv('./steam-dataset/categories.csv', na_values="\\N")
df_genres = pd.read_csv("./steam-dataset/genres.csv", na_values="\\N")
df_reviews = pd.read_csv("./steam-dataset/reviews.csv", na_values="\\N")
df_insights = pd.read_csv("./steam-dataset/steamspy_insights.csv", encoding="ISO-8859-1", na_values="\\N")
df_tags = pd.read_csv("./steam-dataset/tags.csv", na_values="\\N")



df_genres["genre"].value_counts()

df_reviews.dropna(thresh=1)
df_reviews_important = df_reviews.drop(['metacritic_score', 'reviews', 'recommendations', 'steamspy_user_score', 'steamspy_score_rank', 'steamspy_positive', 'steamspy_negative'], axis=1)

categories = df_reviews_important['review_score_description'].unique()
df_reviews_important.dropna(subset=['review_score_description'], inplace=True)
categories = df_reviews_important['review_score_description'].unique()


df_insights_clean = df_insights.drop(['developer', 'publisher', 'price', 'initial_price', 'discount', 'languages', 'genres', 'playtime_average_forever', 'playtime_average_2weeks', 'playtime_median_forever', 'playtime_median_2weeks'], axis=1)
df_insights_clean

df_tags_count = df_tags["tag"].value_counts()

df_games = df_games.drop(["price", "currency", "is_free"], axis=1)

games = df_games.merge(df_reviews_important.set_index("app_id"), on="app_id", how="inner")
games = games.merge(df_insights_clean.set_index("app_id"), on="app_id", how="inner")
games

def standardize_data(df):

	'''
	This function standardize an array, its substracts mean value,
	and then divide the standard deviation.

	param 1: array
	return: standardized array
	'''

	mean = df.mean()
	std = df.std()
	new_df = (df - mean) / std

	return new_df, mean, std

games_one_hot = one_hot_encoding(games, 'owners_range')
games_one_hot = one_hot_encoding(games_one_hot, 'review_score_description')

games_final = games_one_hot.drop(columns=['app_id', 'name', 'release_date'])
games_final = games_final.sample(frac=1)

# %%
cols_x_ordenadas = [
    "price_eur",
    "review_score",
    "positive",
    "negative",
    "total",
    "concurrent_users_yesterday",
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

cols_y_ordenadas = [
    "Overwhelmingly Positive",
    "Very Positive",
    "Mostly Positive",
    "Positive",
    "Mixed",
    "Mostly Negative",
    "Negative",
    "Very Negative",
    "Overwhelmingly Negative",
    "No user reviews",
]

df_x = games_final.reindex(columns=cols_x_ordenadas)
df_y = games_final.reindex(columns=cols_y_ordenadas)

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

display(df_x_train)
display(df_y_train)

# %%
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# %%
rnd_clf = RandomForestClassifier(n_estimators=400, max_leaf_nodes=10, n_jobs=-1, random_state=42)
rnd_clf.fit(df_x_train, df_y_train)
y_pred_rf= rnd_clf.predict(df_x_test)
print("Accuracy", accuracy_score(df_y_test, y_pred_rf))
rnd_clf

# %%
y_pred_raw = rnd_clf.predict(df_x_test[:10])
y_test_raw = df_y_test.iloc[:10].to_numpy()

class_names = df_y_test.columns.tolist()

actual_indices = np.argmax(y_test_raw, axis=1)
predicted_indices = np.argmax(y_pred_raw, axis=1)

actual_labels = [class_names[idx] for idx in actual_indices]
predicted_labels = [class_names[idx] for idx in predicted_indices]

results_rf = pd.DataFrame(
    {
        "Clase Esperada": actual_labels,
        "Clase Predicha": predicted_labels,
        "¿Correcto?": np.array(actual_labels) == np.array(predicted_labels),
    }
)

display(results_rf)

# %%
def capturar_datos_juego():
    print("=== CAPTURA DE DATOS DEL JUEGO DE STEAM ===")

    price_eur = float(input("Precio en EUR (ej. 19.99): "))
    review_score = float(input("Calificación (ej. 8.4): "))
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
    # grade = [
    #     "Overwhelmingly Positive",
    #     "Very Positive",
    #     "Mostly Positive",
    #     "Positive",
    #     "Mixed",
    #     "Negative",
    #     "Mostly Negative",
    #     "Very Negative",
    #     "Overwhelmingly Negative",
    #     "No user reviews",
    # ]

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

    # print("\n--- Selecciona la calificación de acuerdo a reseñas ---")
    # for i, rango in enumerate(grade, 1):
    #     print(f"[{i}] {rango}")

    # while True:
    #     try:
    #         opcion = int(input(f"Elige una opción (1-{len(grade)}): "))
    #         if 1 <= opcion <= len(grade):
    #             grade_selected_idx = opcion - 1
    #             break
    #         else:
    #             print("Opción fuera de rango. Intenta de nuevo.")
    #     except ValueError:
    #         print("Por favor, ingresa solo un número entero.")

    datos_capturados = {
        "price_eur": price_eur,
        "review_score": review_score,
        "positive": positive,
        "negative": negative,
        "total": total,
        "concurrent_users_yesterday": concurrent_users,
    }

    for idx, rango in enumerate(rangos_owners):
        column_name = rango
        datos_capturados[column_name] = 1 if idx == owners_selected_idx else 0

    # for idx, g in enumerate(grade):
    #     column_name = f"grade_{g}"
    #     datos_capturados[column_name] = 1 if idx == grade_selected_idx else 0

    print("\n¡Datos capturados y procesados exitosamente!")
    return pd.DataFrame([datos_capturados])

# %%
entradas = capturar_datos_juego()
print(entradas)
prediction = rnd_clf.predict(entradas)

class_names = df_y_test.columns.tolist()
predicted_indices = np.argmax(prediction, axis=1)
predicted_labels = [class_names[idx] for idx in predicted_indices]

print("Valor predecido: ", predicted_labels)


