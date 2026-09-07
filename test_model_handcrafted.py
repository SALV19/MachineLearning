import pandas as pd
import numpy as np
import json

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

def capturar_datos_juego():
    print("=== CAPTURA DE DATOS DEL JUEGO DE STEAM ===")

    price_eur = float(input("Precio en EUR (ej. 19.99): "))
    # review_score = float(
    #     input("Review score o Calificación (ej. 8 o 8.5): ")
    # )
    positive = int(input("Cantidad de reseñas positivas (ej. 15000): "))
    negative = int(input("Cantidad de reseñas negativas (ej. 1200): "))
    total = int(
        input("Cantidad total de reseñas (ej. 16200): ")
        if positive is None
        else positive + negative
    )
    concurrent_users = int(
        input("Usuarios concurrentes ayer (ej. 4500): ")
    )

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
    grade = [
        "Overwhelmingly Positive",
        "Very Positive",
        "Mostly Positive",
        "Positive",
        "Mixed",
        "Negative",
        "Mostly Negative",	
        "Very Negative"
        "Overwhelmingly Negative",
        "No user reviews",
    ]

    print("\n--- Selecciona el rango de propietarios (Owners Range) ---")
    for i, rango in enumerate(rangos_owners, 1):
        print(f"[{i}] {rango}")

    # Validación de selección única
    while True:
        try:
            opcion = int(
                input(f"Elige una opción (1-{len(rangos_owners)}): ")
            )
            if 1 <= opcion <= len(rangos_owners):
                owners_range_selected = rangos_owners[opcion - 1]
                break
            else:
                print("Opción fuera de rango. Intenta de nuevo.")
        except ValueError:
            print("Por favor, ingresa solo un número entero.")
            
    print("\n--- Selecciona la calificación de acuerdo a reseñas ---")
    for i, rango in enumerate(grade, 1):
        print(f"[{i}] {rango}")

    # Validación de selección única
    while True:
        try:
            opcion = int(
                input(f"Elige una opción (1-{len(grade)}): ")
            )
            if 1 <= opcion <= len(grade):
                grade_selection = grade[opcion - 1]
                break
            else:
                print("Opción fuera de rango. Intenta de nuevo.")
        except ValueError:
            print("Por favor, ingresa solo un número entero.")

    # Guardar en un diccionario / DataFrame
    datos_capturados = {
        "price_eur": price_eur,
        # "review_score": review_score,
        "positive": positive,
        "negative": negative,
        "total": total,
        "concurrent_users_yesterday": concurrent_users,
        "owners_range": owners_range_selected,
        "grade": grade_selection,
    }

    print("\n¡Datos capturados exitosamente!")
    return pd.DataFrame([datos_capturados])

with open("params-10_000.json", 'r') as file:
  params = json.load(file)
  entradas = capturar_datos_juego()
  print(entradas)
  prediction = h(params[:-1], entradas, params[-1])
  
  print("Valor predecido: ", prediction)