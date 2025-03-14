import numpy as np
from libs.auxiliaries_functions import create_functions_json, get_image_paths
# ---------------------------
# Parámetros del algoritmo
# ---------------------------
POP_SIZE = 100            # Número de individuos en la población
NUM_GENERATIONS = 50     # Número de generaciones
NUM_RUNS = 5             # Número de ejecuciones completas (ciclos)

# Parámetros de la función de Langermann
a = np.array([3, 5, 2, 1, 7])
b = np.array([5, 2, 1, 4, 9])
c = np.array([1, 2, 5, 2, 3])

# Parámetros del torneo
TOURNAMENT_SIZE = 3  # Número de individuos participantes en cada torneo

# Parámetros del cruzamiento SBX
CROSSOVER_PROB = 0.9  # Probabilidad de aplicar cruzamiento
ETA_C = 0.5            # Índice de distribución para SBX

# Limites superiores e inferiores
LB = np.array([0.0, 0.0])
UB = np.array([10.0, 1.0])

# Parámetros de la mutación polinomial
MUTATION_PROB = 10.0 / 2  # Probabilidad de mutar cada gen
ETA_MUT = 0.5                         # Índice de distribución para mutación polinomial

# Ruta a la imagen a contrastar
IMG_PATH = get_image_paths('imgs')

best_solutions_list = [] 
all_runs_history = []  # Para graficar luego

FUNCTIONS = create_functions_json(IMG_PATH, LB, UB, NUM_RUNS)