# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


import numpy as np

def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n < 2:
        return None
    
    wezly = []
    for k in range(n): # k = 0, 1, ..., n-1
        xk = np.cos(k * np.pi / (n - 1)) 
        wezly.append(xk)
        
    nodes = np.array(wezly)
    return nodes


def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    wi = []
    for j in range(n):
        if j == 0 or j == n-1:
            wj = 0.5 * (-1)**j
        else:
            wj = (-1)**j
        wi.append(wj)
    return np.array(wi)



def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).
    """

    if not isinstance(xi, np.ndarray) or not isinstance(yi, np.ndarray) or \
       not isinstance(wi, np.ndarray) or not isinstance(x, np.ndarray):
        return None
    
    if not (xi.shape == yi.shape and xi.shape == wi.shape and xi.ndim == 1):
        return None 

    x = x.flatten()

    interpolated_values = np.zeros_like(x, dtype=float)

    for j, point_x in enumerate(x):

        diff = point_x - xi

        is_node = (np.abs(diff) < 1e-12)
        
        if np.any(is_node):

            node_index = np.where(is_node)[0][0]
            interpolated_values[j] = yi[node_index]
        else:

            w_over_diff = wi / diff

            numerator = np.sum(w_over_diff * yi)

            denominator = np.sum(w_over_diff)

            if denominator != 0:
                interpolated_values[j] = numerator / denominator
            else:

                interpolated_values[j] = np.nan 

    return interpolated_values


def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    try:
        xr_arr = np.asarray(xr, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        
        if xr_arr.shape != x_arr.shape:
             if xr_arr.ndim > 0 and x_arr.ndim > 0 and xr_arr.size != x_arr.size:
                return None
            
        max_abs_diff = np.max(np.abs(xr_arr - x_arr))
        
        return float(max_abs_diff)
        
    except Exception:
        return None
