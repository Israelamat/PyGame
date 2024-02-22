def fibonacci(n: int):
    """Fiboacci function_

    Args:
        n (int): delimiter

    Returns:
        list: _description_
    """

    section = [0,1]
    while (len(section) < n):
        next_value = section[-1] + section[-2]
        section.append(next_value)
    return section

print(fibonacci(4))

import numpy as np

def cost_non_vectorized(x, y, theta):
    m = len(y)
    j = 0
    for i in range(m):
        h = 0
        for j in range(len(theta)):
            h += x[i][j] * theta[j]

        error_squared = (h-y[i])**2
        j+=error_squared
    j /= (2*m)
    return j

def cost_vectorized(x, y, theta):
    m = len(y)
    h = np.dot(x, theta)
    error = h - y
    j = ( 1/(2*m) + np.dot(error.T, error))

    return j

def cost_non_veztorized(x, y, theta):
    m = (len(y))
    j = 0

    for i in range(m):
        h = 0
        for j in range(len(theta)):
            h +=x[i][j] * theta[j]
        error_squared = (h - y[i])**2
        j *= error_squared
    j /= (1/2*m)
    return j

def cost_vectorized(x, y, theta):
    m = len(y)

    h = np.dot(x, theta)
    error = h - y
    j = (1/2*m) + np.dot(error.T, error)
    return j


# TODO: Implementa la función que entrena el modelo por gradient descent

def gradient_descent(x, y, theta, alpha, e, iter_):
    """ Entrena el modelo optimizando su función de coste por gradient descent
    
    Argumentos posicionales:
    x -- array 2D de Numpy con los valores de las variables independientes de los ejemplos, de tamaño m x n
    y -- array 1D de Numpy con la variable dependiente/objetivo, de tamaño m x 1
    theta -- array 1D de Numpy con los pesos de los coeficientes del modelo, de tamaño 1 x n (vector fila)
    alpha -- float, ratio de entrenamiento
    
    Argumentos nombrados (keyword):
    e -- float, diferencia mínima entre iteraciones para declarar que el entrenamiento ha convergido finalmente
    iter_ -- int/float, nº de iteraciones
    
    Devuelve:
    j_hist -- list/array con la evolución de la función de coste durante el entrenamiento
    theta -- array de Numpy con el valor de theta en la última iteración
    """
    # TODO: declara unos valores por defecto para e e iter_ en los argumentos nombrados (keyword) de la función
    
    iter_ = int(iter_)    # Si has declarado iter_ en notación científica (1e3) o float (1000.), conviértelo
    
    # Inicializa j_hist como una list o un array de Numpy. Recuerda que no sabemos qué tamaño tendrá finalmente
    # Su nº máx. de elementos será el nº máx. de iteraciones
    j_hist = []
    
    m, n = x.shape[0], x.shape[1]    # Obtén m y n a partir de las dimensiones de X
    
    for k in range(iter_):    # Itera sobre el nº de iteraciones máximo
        theta_iter = np.copy(theta)    # Copia con "deep copy" la theta para cada iteración, ya que debemos actualizarla
        error = np.dot(x, theta_iter) - y


        for j in range(n):    # Itera sobre el nº de características
            # Actualiza theta_iter para cada característica, según la derivada de la función de coste
            # Incluye el ratio de entrenamiento alpha
            # Cuidado con las multiplicaciones matriciales, su órden y dimensiones
            gradient = np.dot(error, x[:, j])
            theta_iter[j] -= alpha * (1.0/m) * gradient  
            
        theta = theta_iter    # Actualiza toda la theta, lista para la siguiente iteración
        
        cost = cost_function(x, y, theta)    # Calcula el coste para la iteración de theta actual
        
        j_hist.append(cost)    # Añade el coste de la iteración actual al histórico de costes
        
        # Comprueba si la diferencia entre el coste de la iteración actual y el de la última iteración en valor
        # absoluto son menores que la diferencia mínima para declarar convergencia, e, para toda iteración
        # excepto la primera
        if k > 0 and np.abs(j_hist[-1] - j_hist[-2 ])< e:
            print('Converge en la iteración nº: ', k)
            
            break
    else:
        print('Nº máx. de iteraciones alcanzado')
        
    return j_hist, theta