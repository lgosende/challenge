# Importar bibliotecas necesarias
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Librerías de modelado
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.model_selection import cross_validate
import scipy.stats as st

import matplotlib.pyplot as plt

# Definir función para evaluar modelos
def evaluar_modelo(y_true, y_pred, nombre_modelo):
    """
    Evalúa un modelo usando múltiples métricas
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    resultados = {
        'Modelo': nombre_modelo,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    print(f"\n{nombre_modelo}:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return resultados

def evaluar_modelo_con_ic(y_true, y_pred, nombre_modelo, X_test=None, modelo=None, alpha=0.05):
    """
    Evalúa un modelo usando múltiples métricas e incluye intervalos de confianza del 95%
    
    Parameters:
    -----------
    y_true : array-like
        Valores reales
    y_pred : array-like  
        Predicciones del modelo
    nombre_modelo : str
        Nombre del modelo
    X_test : array-like, optional
        Datos de test para bootstrap
    modelo : object, optional
        Modelo entrenado para bootstrap
    alpha : float
        Nivel de significancia (0.05 para 95% de confianza)
    """
    # Métricas puntuales
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calcular intervalos de confianza usando bootstrap
    n_bootstrap = 1000
    bootstrap_rmse = []
    bootstrap_mae = []
    bootstrap_r2 = []
    
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        # Muestreo bootstrap
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = np.array(y_true)[bootstrap_idx]
        y_pred_boot = np.array(y_pred)[bootstrap_idx]
        
        # Calcular métricas para esta muestra bootstrap
        rmse_boot = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
        mae_boot = mean_absolute_error(y_true_boot, y_pred_boot)
        r2_boot = r2_score(y_true_boot, y_pred_boot)
        
        bootstrap_rmse.append(rmse_boot)
        bootstrap_mae.append(mae_boot)
        bootstrap_r2.append(r2_boot)
    
    # Calcular intervalos de confianza (percentiles)
    confidence_level = (1 - alpha) * 100
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    rmse_ci = np.percentile(bootstrap_rmse, [lower_percentile, upper_percentile])
    mae_ci = np.percentile(bootstrap_mae, [lower_percentile, upper_percentile])
    r2_ci = np.percentile(bootstrap_r2, [lower_percentile, upper_percentile])
    
    resultados = {
        'Modelo': nombre_modelo,
        'RMSE': rmse,
        'RMSE_CI_Lower': rmse_ci[0],
        'RMSE_CI_Upper': rmse_ci[1],
        'MAE': mae,
        'MAE_CI_Lower': mae_ci[0],
        'MAE_CI_Upper': mae_ci[1],
        'R²': r2,
        'R²_CI_Lower': r2_ci[0],
        'R²_CI_Upper': r2_ci[1]
    }
    
    print(f"\n{nombre_modelo}:")
    print(f"  RMSE: ${rmse:,.2f} (95% IC: ${rmse_ci[0]:,.2f} - ${rmse_ci[1]:,.2f})")
    print(f"  MAE:  ${mae:,.2f} (95% IC: ${mae_ci[0]:,.2f} - ${mae_ci[1]:,.2f})")
    print(f"  R²:   {r2:.4f} (95% IC: {r2_ci[0]:.4f} - {r2_ci[1]:.4f})")
    
    return resultados

def cross_validation_con_ic(modelo, X, y, cv=5, scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']):
    """
    Realiza validación cruzada y calcula intervalos de confianza para las métricas
    """
    # Realizar validación cruzada
    cv_results = cross_validate(modelo, X, y, cv=cv, scoring=scoring, return_train_score=False)
    
    resultados_ic = {}
    
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        
        # Para métricas negativas, convertir a positivas
        if 'neg_' in metric:
            scores = -scores
            metric_name = metric.replace('neg_', '').replace('_', ' ').title()
        else:
            metric_name = metric.replace('_', ' ').title()
        
        # Calcular estadísticas
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Intervalo de confianza usando distribución t de Student
        # (más apropiado para muestras pequeñas)
        confidence = 0.95
        degrees_freedom = cv - 1
        t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
        margin_error = t_value * (std_score / np.sqrt(cv))
        
        ci_lower = mean_score - margin_error
        ci_upper = mean_score + margin_error
        
        resultados_ic[metric_name] = {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'scores': scores
        }
        
        print(f"{metric_name}:")
        print(f"  Media: {mean_score:.4f} ± {std_score:.4f}")
        print(f"  95% IC: {ci_lower:.4f} - {ci_upper:.4f}")
        print(f"  Scores individuales: {scores}")
        print()
    
    return resultados_ic

# Intervalos de confianza para los coeficientes usando estadísticas bootstrapped
def calcular_ic_coeficientes(modelo_simple, X, y, n_bootstrap=1000, alpha=0.05):

    """Calcula intervalos de confianza para coeficientes usando bootstrap"""
    coef_bootstrap = []
    intercept_bootstrap = []
    
    n_samples = len(y)
    
    for i in range(n_bootstrap):
        # Muestreo bootstrap
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[bootstrap_idx] if hasattr(X, 'iloc') else X[bootstrap_idx]
        y_boot = y.iloc[bootstrap_idx] if hasattr(y, 'iloc') else y[bootstrap_idx]
        
        # Entrenar modelo en muestra bootstrap
        modelo_boot = LinearRegression()
        modelo_boot.fit(X_boot, y_boot)
        
        coef_bootstrap.append(modelo_boot.coef_[0])
        intercept_bootstrap.append(modelo_boot.intercept_)
    
    # Calcular intervalos de confianza
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    coef_ci = np.percentile(coef_bootstrap, [lower_percentile, upper_percentile])
    intercept_ci = np.percentile(intercept_bootstrap, [lower_percentile, upper_percentile])
    
    print(f"\nIntervalos de confianza del 95% para coeficientes:")
    print(f"  Intercepto: ${modelo_simple.intercept_:,.2f} (IC: ${intercept_ci[0]:,.2f} - ${intercept_ci[1]:,.2f})")
    print(f"  Pendiente: ${modelo_simple.coef_[0]:,.2f} (IC: ${coef_ci[0]:,.2f} - ${coef_ci[1]:,.2f})")
    
    return {
        'intercept': {'value': modelo_simple.intercept_, 'ci': intercept_ci},
        'coef': {'value': modelo_simple.coef_[0], 'ci': coef_ci}
    }

def calcular_ic_prediccion(modelo, X_train, y_train, X_nuevo, alpha=0.05):
    """
    Calcula intervalos de confianza e intervalos de predicción para regresión lineal
    
    Parameters:
    -----------
    modelo : LinearRegression
        Modelo entrenado
    X_train : array-like
        Datos de entrenamiento
    y_train : array-like  
        Target de entrenamiento
    X_nuevo : array-like
        Nuevos puntos para predecir
    alpha : float
        Nivel de significancia (0.05 para 95%)
    
    Returns:
    --------
    dict con predicciones e intervalos
    """
    # Predicciones puntuales
    y_pred = modelo.predict(X_nuevo.reshape(-1, 1))
    
    # Calcular parámetros estadísticos
    n = len(X_train)
    k = X_train.shape[1]  # número de predictores
    
    # Residuos y MSE
    y_train_pred = modelo.predict(X_train)
    residuos = y_train - y_train_pred
    mse = np.mean(residuos**2)
    
    # Grados de libertad
    df = n - k - 1
    
    # Valor t crítico
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Matriz de diseño para nuevos puntos
    X_train_mean = np.mean(X_train)
    X_train_var = np.var(X_train, ddof=1)
    
    # Calcular intervalos
    se_pred = []
    se_conf = []
    
    for x_val in X_nuevo:
        # Error estándar para intervalo de confianza (media)
        se_conf_i = np.sqrt(mse * (1/n + (x_val - X_train_mean)**2 / ((n-1) * X_train_var)))
        se_conf.append(se_conf_i)
        
        # Error estándar para intervalo de predicción (nueva observación)
        se_pred_i = np.sqrt(mse * (1 + 1/n + (x_val - X_train_mean)**2 / ((n-1) * X_train_var)))
        se_pred.append(se_pred_i)
    
    se_conf = np.array(se_conf)
    se_pred = np.array(se_pred)
    
    # Intervalos de confianza y predicción
    ic_conf_lower = y_pred - t_crit * se_conf
    ic_conf_upper = y_pred + t_crit * se_conf
    
    ic_pred_lower = y_pred - t_crit * se_pred
    ic_pred_upper = y_pred + t_crit * se_pred
    
    return {
        'prediccion': y_pred,
        'ic_confianza_lower': ic_conf_lower,
        'ic_confianza_upper': ic_conf_upper,
        'ic_prediccion_lower': ic_pred_lower,
        'ic_prediccion_upper': ic_pred_upper
    }

# =============================================================================
# GRÁFICO PRINCIPAL CON INTERVALOS DE CONFIANZA
# =============================================================================

def plot_regresion_con_ic(modelo, X_train, y_train, X_test, y_test, X_simple):
    """
    Crea visualización completa de regresión con intervalos de confianza
    """
    
    plt.figure(figsize=(15, 10))
    
    # =========================================================================
    # SUBPLOT 1: REGRESIÓN CON INTERVALOS DE CONFIANZA
    # =========================================================================
    plt.subplot(2, 2, 1)
    
    # Datos de entrenamiento y prueba
    plt.scatter(X_train, y_train, alpha=0.6, label='Entrenamiento', color='blue', s=50)
    plt.scatter(X_test, y_test, alpha=0.6, label='Prueba', color='red', s=50)
    
    # Rango para la línea de regresión
    x_range = np.linspace(X_simple.min().values[0], X_simple.max().values[0], 100)
    
    # Calcular intervalos de confianza
    intervalos = calcular_ic_prediccion(modelo, X_train.values.flatten(), y_train.values, x_range)
    
    # Línea de regresión
    plt.plot(x_range, intervalos['prediccion'], color='green', linewidth=3, 
             label='Regresión Lineal', zorder=5)
    
    # Intervalo de confianza (95% para la media)
    plt.fill_between(x_range, 
                     intervalos['ic_confianza_lower'], 
                     intervalos['ic_confianza_upper'],
                     alpha=0.2, color='green', 
                     label='IC 95% (Media)', zorder=3)
    
    # Intervalo de predicción (95% para nuevas observaciones)
    plt.fill_between(x_range, 
                     intervalos['ic_prediccion_lower'], 
                     intervalos['ic_prediccion_upper'],
                     alpha=0.1, color='orange', 
                     label='IC 95% (Predicción)', zorder=2)
    
    plt.xlabel('Años de Experiencia')
    plt.ylabel('Salario ($)')
    plt.title('Regresión Lineal con Intervalos de Confianza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Formatear eje Y para mostrar valores en miles
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # =========================================================================
    # SUBPLOT 2: ZOOM EN REGIÓN CENTRAL
    # =========================================================================
    plt.subplot(2, 2, 2)
    
    # Foco en la región central (donde hay más datos)
    x_central = np.linspace(np.percentile(X_simple, 25), np.percentile(X_simple, 75), 50)
    intervalos_central = calcular_ic_prediccion(modelo, X_train.values.flatten(), y_train.values, x_central)
    
    # Filtrar datos en región central
    mask_train = (X_train.values.flatten() >= x_central.min()) & (X_train.values.flatten() <= x_central.max())
    mask_test = (X_test.values.flatten() >= x_central.min()) & (X_test.values.flatten() <= x_central.max())
    
    plt.scatter(X_train[mask_train], y_train[mask_train], alpha=0.7, color='blue', s=60)
    plt.scatter(X_test[mask_test], y_test[mask_test], alpha=0.7, color='red', s=60)
    
    plt.plot(x_central, intervalos_central['prediccion'], color='green', linewidth=3)
    plt.fill_between(x_central, 
                     intervalos_central['ic_confianza_lower'], 
                     intervalos_central['ic_confianza_upper'],
                     alpha=0.3, color='green')
    plt.fill_between(x_central, 
                     intervalos_central['ic_prediccion_lower'], 
                     intervalos_central['ic_prediccion_upper'],
                     alpha=0.15, color='orange')
    
    plt.xlabel('Años de Experiencia')
    plt.ylabel('Salario ($)')
    plt.title('Zoom: Región Central con IC')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # =========================================================================
    # SUBPLOT 3: RESIDUOS VS PREDICCIONES CON IC
    # =========================================================================
    plt.subplot(2, 2, 3)
    
    y_pred_test = modelo.predict(X_test)
    residuos_test = y_test - y_pred_test
    
    plt.scatter(y_pred_test, residuos_test, alpha=0.6, color='purple', s=50)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Calcular límites de residuos (±2 desviaciones estándar)
    std_residuos = np.std(residuos_test)
    plt.axhline(y=2*std_residuos, color='orange', linestyle=':', alpha=0.7, label='+2σ')
    plt.axhline(y=-2*std_residuos, color='orange', linestyle=':', alpha=0.7, label='-2σ')
    
    # Banda de confianza para residuos
    plt.fill_between(y_pred_test.min(), y_pred_test.max(), 
                     -2*std_residuos, 2*std_residuos, 
                     alpha=0.1, color='orange')
    
    plt.xlabel('Predicciones ($)')
    plt.ylabel('Residuos ($)')
    plt.title('Residuos vs Predicciones')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # =========================================================================
    # SUBPLOT 4: DISTRIBUCIÓN DE RESIDUOS CON ESTADÍSTICAS
    # =========================================================================
    plt.subplot(2, 2, 4)
    
    # Histograma de residuos
    n, bins, patches = plt.hist(residuos_test, bins=15, alpha=0.7, color='skyblue', 
                               density=True, edgecolor='black', linewidth=0.5)
    
    # Curva normal teórica
    mu, sigma = np.mean(residuos_test), np.std(residuos_test)
    x_norm = np.linspace(residuos_test.min(), residuos_test.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    plt.plot(x_norm, y_norm, 'red', linewidth=2, label=f'Normal(μ={mu:.0f}, σ={sigma:.0f})')
    
    # Líneas de referencia
    plt.axvline(mu, color='red', linestyle='--', alpha=0.7, label=f'Media: ${mu:.0f}')
    plt.axvline(mu + sigma, color='orange', linestyle=':', alpha=0.7)
    plt.axvline(mu - sigma, color='orange', linestyle=':', alpha=0.7)
    
    plt.xlabel('Residuos ($)')
    plt.ylabel('Densidad')
    plt.title('Distribución de Residuos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # =========================================================================
    # ESTADÍSTICAS RESUMIDAS
    # =========================================================================
    print("=" * 60)
    print("           ESTADÍSTICAS DE INTERVALOS DE CONFIANZA")
    print("=" * 60)
    
    # Ancho promedio de intervalos
    ancho_ic_conf = np.mean(intervalos['ic_confianza_upper'] - intervalos['ic_confianza_lower'])
    ancho_ic_pred = np.mean(intervalos['ic_prediccion_upper'] - intervalos['ic_prediccion_lower'])
    
    print(f"Ancho promedio IC confianza (media): ${ancho_ic_conf:,.0f}")
    print(f"Ancho promedio IC predicción: ${ancho_ic_pred:,.0f}")
    print(f"Ratio IC predicción / IC confianza: {ancho_ic_pred/ancho_ic_conf:.1f}x")
    
    # Estadísticas de residuos
    print(f"\nEstadísticas de residuos:")
    print(f"  Media: ${np.mean(residuos_test):,.0f}")
    print(f"  Desviación estándar: ${np.std(residuos_test):,.0f}")
    print(f"  Rango: ${residuos_test.min():,.0f} a ${residuos_test.max():,.0f}")
    
    # Test de normalidad de residuos
    from scipy.stats import shapiro
    stat, p_value = shapiro(residuos_test)
    print(f"\nTest de normalidad de residuos (Shapiro-Wilk):")
    print(f"  Estadístico: {stat:.4f}")
    print(f"  p-valor: {p_value:.4f}")
    print(f"  Normalidad: {'✅ Sí' if p_value > 0.05 else '❌ No'} (α=0.05)")
    
    print("=" * 60)

def plot_regresion_simple_con_ic(modelo, X_train, y_train, X_test, y_test, X_simple):
    """
    Versión simplificada del gráfico con intervalos de confianza
    """
    plt.figure(figsize=(12, 8))
    
    # Datos
    plt.scatter(X_train, y_train, alpha=0.6, label='Entrenamiento', color='blue')
    plt.scatter(X_test, y_test, alpha=0.6, label='Prueba', color='red')
    
    # Rango para predicciones
    x_range = np.linspace(X_simple.min().values[0], X_simple.max().values[0], 100)
    
    # Calcular intervalos
    intervalos = calcular_ic_prediccion(modelo, X_train.values.flatten(), y_train.values, x_range)
    
    # Línea de regresión
    plt.plot(x_range, intervalos['prediccion'], color='green', linewidth=2, label='Regresión Lineal')
    
    # Intervalos de confianza
    plt.fill_between(x_range, 
                     intervalos['ic_confianza_lower'], 
                     intervalos['ic_confianza_upper'],
                     alpha=0.2, color='green', label='IC 95% (Media)')
    
    plt.fill_between(x_range, 
                     intervalos['ic_prediccion_lower'], 
                     intervalos['ic_prediccion_upper'],
                     alpha=0.1, color='orange', label='IC 95% (Predicción)')
    
    plt.xlabel('Años de Experiencia')
    plt.ylabel('Salario ($)')
    plt.title('Regresión Lineal Simple con Intervalos de Confianza del 95%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.show()

def calcular_ic_prediccion_corregida(modelo, X_train, y_train, X_nuevo, alpha=0.05):
    """
    Calcula intervalos de confianza e intervalos de predicción para regresión lineal
    VERSIÓN CORREGIDA para manejar arrays 1D y 2D
    """
    # Asegurar que los datos estén en el formato correcto
    if hasattr(X_train, 'values'):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
    else:
        y_train_array = y_train
    
    # Asegurar que X_train sea 2D
    if X_train_array.ndim == 1:
        X_train_array = X_train_array.reshape(-1, 1)
    
    # Asegurar que X_nuevo sea 2D
    if X_nuevo.ndim == 1:
        X_nuevo_2d = X_nuevo.reshape(-1, 1)
    else:
        X_nuevo_2d = X_nuevo
    
    # Predicciones puntuales
    y_pred = modelo.predict(X_nuevo_2d)
    
    # Calcular parámetros estadísticos
    n = len(X_train_array)
    k = X_train_array.shape[1]  # número de predictores (ahora debería funcionar)
    
    # Residuos y MSE
    y_train_pred = modelo.predict(X_train_array)
    residuos = y_train_array - y_train_pred
    mse = np.mean(residuos**2)
    
    # Grados de libertad
    df = n - k - 1
    
    # Valor t crítico
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Para regresión simple (1 variable), usar cálculo simplificado
    if k == 1:
        X_train_flat = X_train_array.flatten()
        X_train_mean = np.mean(X_train_flat)
        X_train_sse = np.sum((X_train_flat - X_train_mean)**2)
        
        # Calcular intervalos para cada punto
        se_conf = []
        se_pred = []
        
        for x_val in X_nuevo:
            # Error estándar para intervalo de confianza (media)
            se_conf_i = np.sqrt(mse * (1/n + (x_val - X_train_mean)**2 / X_train_sse))
            se_conf.append(se_conf_i)
            
            # Error estándar para intervalo de predicción (nueva observación)
            se_pred_i = np.sqrt(mse * (1 + 1/n + (x_val - X_train_mean)**2 / X_train_sse))
            se_pred.append(se_pred_i)
        
        se_conf = np.array(se_conf)
        se_pred = np.array(se_pred)
    
    else:
        # Para regresión múltiple (más complejo)
        # Usar aproximación simplificada
        se_mean = np.sqrt(mse / n)
        se_conf = np.full(len(X_nuevo), se_mean)
        se_pred = np.sqrt(mse + se_mean**2)
        se_pred = np.full(len(X_nuevo), se_pred)
    
    # Intervalos de confianza y predicción
    ic_conf_lower = y_pred - t_crit * se_conf
    ic_conf_upper = y_pred + t_crit * se_conf
    
    ic_pred_lower = y_pred - t_crit * se_pred
    ic_pred_upper = y_pred + t_crit * se_pred
    
    return {
        'prediccion': y_pred,
        'ic_confianza_lower': ic_conf_lower,
        'ic_confianza_upper': ic_conf_upper,
        'ic_prediccion_lower': ic_pred_lower,
        'ic_prediccion_upper': ic_pred_upper
    }

# =============================================================================
# VERSIÓN SIMPLIFICADA CORREGIDA
# =============================================================================

def plot_regresion_simple_con_ic_corregida(modelo, X_train, y_train, X_test, y_test, X_simple):
    """
    Versión simplificada y CORREGIDA del gráfico con intervalos de confianza
    """
    plt.figure(figsize=(12, 8))
    
    # Datos
    plt.scatter(X_train, y_train, alpha=0.6, label='Entrenamiento', color='blue', s=50)
    plt.scatter(X_test, y_test, alpha=0.6, label='Prueba', color='red', s=50)
    
    # Rango para predicciones
    x_min = X_simple.min().values[0] if hasattr(X_simple.min(), 'values') else X_simple.min()
    x_max = X_simple.max().values[0] if hasattr(X_simple.max(), 'values') else X_simple.max()
    x_range = np.linspace(x_min, x_max, 100)
    
    # Calcular intervalos usando la función corregida
    intervalos = calcular_ic_prediccion_corregida(modelo, X_train, y_train, x_range)
    
    # Línea de regresión
    plt.plot(x_range, intervalos['prediccion'], color='green', linewidth=2, label='Regresión Lineal')
    
    # Intervalos de confianza
    plt.fill_between(x_range, 
                     intervalos['ic_confianza_lower'], 
                     intervalos['ic_confianza_upper'],
                     alpha=0.2, color='green', label='IC 95% (Media)')
    
    plt.fill_between(x_range, 
                     intervalos['ic_prediccion_lower'], 
                     intervalos['ic_prediccion_upper'],
                     alpha=0.1, color='orange', label='IC 95% (Predicción)')
    
    plt.xlabel('Años de Experiencia')
    plt.ylabel('Salario ($)')
    plt.title('Regresión Lineal Simple con Intervalos de Confianza del 95%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Formatear eje Y
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas adicionales
    ancho_ic_conf = np.mean(intervalos['ic_confianza_upper'] - intervalos['ic_confianza_lower'])
    ancho_ic_pred = np.mean(intervalos['ic_prediccion_upper'] - intervalos['ic_prediccion_lower'])
    
    print("=" * 50)
    print("    ESTADÍSTICAS DE INTERVALOS DE CONFIANZA")
    print("=" * 50)
    print(f"Ancho promedio IC confianza: ${ancho_ic_conf:,.0f}")
    print(f"Ancho promedio IC predicción: ${ancho_ic_pred:,.0f}")
    print(f"Ratio IC predicción / IC confianza: {ancho_ic_pred/ancho_ic_conf:.1f}x")
    
    # Estadísticas del modelo
    y_pred_test = modelo.predict(X_test)
    residuos = y_test.values if hasattr(y_test, 'values') else y_test
    residuos = residuos - y_pred_test
    
    print(f"\nEstadísticas del modelo:")
    print(f"RMSE: ${np.sqrt(np.mean(residuos**2)):,.0f}")
    print(f"Media residuos: ${np.mean(residuos):,.0f}")
    print(f"Std residuos: ${np.std(residuos):,.0f}")
    print("=" * 50)

# =============================================================================
# VERSIÓN SIMPLE
# =============================================================================

def plot_regresion_simple_basico(modelo, X_train, y_train, X_test, y_test, X_simple):
    """
    Versión básica pero robusta del gráfico con intervalos aproximados
    """
    plt.figure(figsize=(12, 8))
    
    # Datos
    plt.scatter(X_train, y_train, alpha=0.6, label='Entrenamiento', color='blue')
    plt.scatter(X_test, y_test, alpha=0.6, label='Prueba', color='red')
    
    # Rango para la línea
    x_min = float(X_simple.min())
    x_max = float(X_simple.max())
    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    
    # Predicciones
    y_pred_range = modelo.predict(x_range)
    
    # Línea de regresión
    plt.plot(x_range.flatten(), y_pred_range, color='green', linewidth=2, label='Regresión Lineal')
    
    # Aproximación simple de intervalos usando RMSE
    y_pred_test = modelo.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred_test)**2))
    
    # Intervalos aproximados (± 1.96 * RMSE para ~95%)
    ic_upper = y_pred_range + 1.96 * rmse
    ic_lower = y_pred_range - 1.96 * rmse
    
    plt.fill_between(x_range.flatten(), ic_lower, ic_upper,
                     alpha=0.2, color='green', label=f'IC Aproximado ±${rmse:,.0f}')
    
    plt.xlabel('Años de Experiencia')
    plt.ylabel('Salario ($)')
    plt.title('Regresión Lineal Simple con Intervalo Aproximado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.show()
    
    print(f"RMSE del modelo: ${rmse:,.0f}")
    print(f"Intervalo aproximado: ±${1.96*rmse:,.0f}")



# =============================================================================
# FUNCIÓN SIMPLIFICADA PARA REGRESIÓN MÚLTIPLE
# =============================================================================

def plot_regresion_multiple_simple(modelo, X_train, y_train, X_test, y_test, feature_names):
    """
    Versión simplificada de visualización para regresión múltiple
    """
    # Predicciones
    y_pred_test = modelo.predict(X_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Predicciones vs Reales
    axes[0].scatter(y_test, y_pred_test, alpha=0.6, color='blue')
    min_val, max_val = min(y_test), max(y_test)
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    axes[0].set_xlabel('Valores Reales ($)')
    axes[0].set_ylabel('Predicciones ($)')
    axes[0].set_title('Predicciones vs Valores Reales')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuos
    residuos = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuos, alpha=0.6, color='green')
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel('Predicciones ($)')
    axes[1].set_ylabel('Residuos ($)')
    axes[1].set_title('Análisis de Residuos')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Importancia de variables
    coeficientes = modelo.coef_
    axes[2].bar(range(len(feature_names)), np.abs(coeficientes), alpha=0.7)
    axes[2].set_xlabel('Variables')
    axes[2].set_ylabel('|Coeficientes|')
    axes[2].set_title('Importancia de Variables')
    axes[2].set_xticks(range(len(feature_names)))
    axes[2].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_regresion_multiple_completa(modelo, X_train, y_train, X_test, y_test, feature_names):
    """
    Visualización completa para regresión múltiple con intervalos de confianza
    VERSIÓN CORREGIDA - sin conflictos de importación
    """
    # Importaciones al inicio de la función para evitar conflictos
    from sklearn.linear_model import LinearRegression as LR
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Calcular métricas
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis de Regresión Múltiple con Intervalos de Confianza', fontsize=16, fontweight='bold')
    
    # =========================================================================
    # 1. PREDICCIONES VS VALORES REALES CON IC
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Scatter plot
    ax1.scatter(y_train, y_pred_train, alpha=0.6, label='Entrenamiento', color='blue', s=50)
    ax1.scatter(y_test, y_pred_test, alpha=0.6, label='Prueba', color='red', s=50)
    
    # Línea diagonal perfecta
    min_val = min(min(y_train), min(y_test))
    max_val = max(max(y_train), max(y_test))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Predicción Perfecta')
    
    # Intervalos de confianza basados en RMSE
    y_range = np.linspace(min_val, max_val, 100)
    ax1.fill_between(y_range, y_range - 1.96*rmse_test, y_range + 1.96*rmse_test,
                     alpha=0.2, color='gray', label=f'IC 95% (±${1.96*rmse_test:,.0f})')
    
    ax1.set_xlabel('Valores Reales ($)')
    ax1.set_ylabel('Predicciones ($)')
    ax1.set_title('Predicciones vs Valores Reales')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # =========================================================================
    # 2. RESIDUOS VS PREDICCIONES CON IC
    # =========================================================================
    ax2 = axes[0, 1]
    
    residuos_train = y_train - y_pred_train
    residuos_test = y_test - y_pred_test
    
    ax2.scatter(y_pred_train, residuos_train, alpha=0.6, color='blue', s=50, label='Entrenamiento')
    ax2.scatter(y_pred_test, residuos_test, alpha=0.6, color='red', s=50, label='Prueba')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    
    # Bandas de confianza para residuos
    std_residuos = np.std(residuos_test)
    ax2.axhline(y=2*std_residuos, color='orange', linestyle=':', alpha=0.7, label='+2σ')
    ax2.axhline(y=-2*std_residuos, color='orange', linestyle=':', alpha=0.7, label='-2σ')
    ax2.fill_between([y_pred_test.min(), y_pred_test.max()], 
                     -2*std_residuos, 2*std_residuos, 
                     alpha=0.1, color='orange')
    
    ax2.set_xlabel('Predicciones ($)')
    ax2.set_ylabel('Residuos ($)')
    ax2.set_title('Residuos vs Predicciones')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # =========================================================================
    # 3. DISTRIBUCIÓN DE RESIDUOS CON TEST DE NORMALIDAD
    # =========================================================================
    ax3 = axes[0, 2]
    
    # Histograma
    n, bins, patches = ax3.hist(residuos_test, bins=15, alpha=0.7, color='skyblue', 
                               density=True, edgecolor='black', linewidth=0.5)
    
    # Curva normal teórica
    mu, sigma = np.mean(residuos_test), np.std(residuos_test)
    x_norm = np.linspace(residuos_test.min(), residuos_test.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    ax3.plot(x_norm, y_norm, 'red', linewidth=2, label=f'Normal(μ={mu:.0f}, σ={sigma:.0f})')
    
    ax3.axvline(mu, color='red', linestyle='--', alpha=0.7, label=f'Media: ${mu:.0f}')
    ax3.set_xlabel('Residuos ($)')
    ax3.set_ylabel('Densidad')
    ax3.set_title('Distribución de Residuos')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # 4. IMPORTANCIA DE VARIABLES CON IC (VERSIÓN SIMPLIFICADA)
    # =========================================================================
    ax4 = axes[1, 0]
    
    # Coeficientes del modelo
    coeficientes = modelo.coef_
    
    # Calcular IC para coeficientes usando bootstrap (VERSIÓN CORREGIDA)
    print("Calculando intervalos de confianza para coeficientes...")
    n_bootstrap = 500  # Reducido para mayor velocidad
    coef_bootstrap = []
    
    # Convertir a arrays si es necesario
    if hasattr(X_train, 'values'):
        X_train_array = X_train.values
        y_train_array = y_train.values
    else:
        X_train_array = X_train
        y_train_array = y_train
    
    for i in range(n_bootstrap):
        # Muestreo bootstrap
        n_samples = len(X_train_array)
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train_array[bootstrap_idx]
        y_boot = y_train_array[bootstrap_idx]
        
        try:
            # Entrenar modelo en muestra bootstrap usando el alias LR
            modelo_boot = LR()
            modelo_boot.fit(X_boot, y_boot)
            coef_bootstrap.append(modelo_boot.coef_)
        except:
            # Si hay error, usar los coeficientes originales
            coef_bootstrap.append(coeficientes)
    
    coef_bootstrap = np.array(coef_bootstrap)
    
    # Calcular IC para cada coeficiente
    coef_lower = np.percentile(coef_bootstrap, 2.5, axis=0)
    coef_upper = np.percentile(coef_bootstrap, 97.5, axis=0)
    
    # Gráfico de barras con intervalos de error
    x_pos = np.arange(len(feature_names))
    yerr_lower = coeficientes - coef_lower
    yerr_upper = coef_upper - coeficientes
    
    ax4.bar(x_pos, coeficientes, alpha=0.7, color='steelblue', 
            yerr=[yerr_lower, yerr_upper],
            capsize=5, error_kw={'alpha': 0.7, 'capthick': 2})
    
    ax4.set_xlabel('Variables')
    ax4.set_ylabel('Coeficientes')
    ax4.set_title('Coeficientes con IC del 95%')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(feature_names, rotation=45, ha='right')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # 5. VARIABLE MÁS IMPORTANTE VS TARGET
    # =========================================================================
    ax5 = axes[1, 1]
    
    # Encontrar la variable más importante (mayor coeficiente absoluto)
    var_mas_importante_idx = np.argmax(np.abs(coeficientes))
    var_mas_importante = feature_names[var_mas_importante_idx]
    
    # Obtener valores de esa variable
    if hasattr(X_train, 'iloc'):
        x_importante_train = X_train.iloc[:, var_mas_importante_idx]
        x_importante_test = X_test.iloc[:, var_mas_importante_idx]
    else:
        x_importante_train = X_train[:, var_mas_importante_idx]
        x_importante_test = X_test[:, var_mas_importante_idx]
    
    # Scatter plot
    ax5.scatter(x_importante_train, y_train, alpha=0.6, color='blue', s=50, label='Entrenamiento')
    ax5.scatter(x_importante_test, y_test, alpha=0.6, color='red', s=50, label='Prueba')
    
    # Línea de tendencia para esta variable específica
    modelo_simple_var = LR()  # Usar el alias LR
    X_var_combined = np.concatenate([x_importante_train, x_importante_test]).reshape(-1, 1)
    y_combined = np.concatenate([y_train, y_test])
    modelo_simple_var.fit(X_var_combined, y_combined)
    
    x_range_var = np.linspace(X_var_combined.min(), X_var_combined.max(), 100).reshape(-1, 1)
    y_pred_var = modelo_simple_var.predict(x_range_var)
    
    ax5.plot(x_range_var.flatten(), y_pred_var, color='green', linewidth=2, 
             label=f'Tendencia (coef={coeficientes[var_mas_importante_idx]:,.0f})')
    
    ax5.set_xlabel(var_mas_importante)
    ax5.set_ylabel('Salario ($)')
    ax5.set_title(f'Variable Más Importante: {var_mas_importante}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # =========================================================================
    # 6. MÉTRICAS Y ESTADÍSTICAS
    # =========================================================================
    ax6 = axes[1, 2]
    ax6.axis('off')  # Quitar ejes para mostrar solo texto
    
    # Test de normalidad
    try:
        stat, p_value = stats.shapiro(residuos_test)
        normalidad_texto = f"p = {p_value:.4f}"
    except:
        normalidad_texto = "No calculado"
    
    # Crear tabla de métricas
    metricas_text = f"""
    MÉTRICAS DEL MODELO
    ═══════════════════════════════
    
    Entrenamiento:
    • R² = {r2_train:.4f}
    • RMSE = ${rmse_train:,.0f}
    
    Prueba:
    • R² = {r2_test:.4f}
    • RMSE = ${rmse_test:,.0f}
    
    Estadísticas Residuos:
    • Media = ${np.mean(residuos_test):,.0f}
    • Std = ${np.std(residuos_test):,.0f}
    • Min = ${residuos_test.min():,.0f}
    • Max = ${residuos_test.max():,.0f}
    
    Variables:
    • Total features: {len(feature_names)}
    • Más importante: {var_mas_importante}
    • Coef. más alto: {coeficientes.max():,.0f}
    • Coef. más bajo: {coeficientes.min():,.0f}
    
    Normalidad residuos: {normalidad_texto}
    """
    
    ax6.text(0.05, 0.95, metricas_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # =========================================================================
    # ESTADÍSTICAS ADICIONALES
    # =========================================================================
    print("=" * 80)
    print("           ANÁLISIS DETALLADO DE REGRESIÓN MÚLTIPLE")
    print("=" * 80)
    
    print(f"\n📊 RENDIMIENTO DEL MODELO:")
    print(f"   • R² Test: {r2_test:.4f}")
    print(f"   • RMSE Test: ${rmse_test:,.0f}")
    overfitting = ((r2_train - r2_test)/r2_train*100) if r2_train > 0 else 0
    print(f"   • Overfitting: {overfitting:+.1f}%")
    
    print(f"\n🔍 ANÁLISIS DE COEFICIENTES:")
    for i, (nombre, coef, lower, upper) in enumerate(zip(feature_names, coeficientes, coef_lower, coef_upper)):
        significativo = "✅" if (lower > 0 and upper > 0) or (lower < 0 and upper < 0) else "❌"
        print(f"   {significativo} {nombre}: {coef:,.0f} (IC: {lower:,.0f} a {upper:,.0f})")
    
    print(f"\n📈 CALIDAD DE RESIDUOS:")
    try:
        stat, p_value = stats.shapiro(residuos_test)
        print(f"   • Test normalidad (Shapiro-Wilk): p = {p_value:.4f}")
        print(f"   • Residuos normales: {'✅ Sí' if p_value > 0.05 else '❌ No'}")
    except:
        print(f"   • Test normalidad: No se pudo calcular")
    
    # Test de heterocedasticidad simple
    try:
        corr_het = np.corrcoef(np.abs(residuos_test), y_pred_test)[0,1]
        print(f"   • Heterocedasticidad: {'⚠️ Posible' if abs(corr_het) > 0.3 else '✅ OK'}")
    except:
        print(f"   • Heterocedasticidad: No evaluada")
    
    print("=" * 80)
    
    return {
        'coeficientes': coeficientes,
        'coef_ic_lower': coef_lower,
        'coef_ic_upper': coef_upper,
        'metricas': {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test
        }
    }

def plot_resultados_con_ic(df_resultados):
    """
    Crea gráficos que muestran las métricas con sus intervalos de confianza
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² con intervalos de confianza
    axes[0, 0].errorbar(
        range(len(df_resultados)), 
        df_resultados['R²'],
        yerr=[df_resultados['R²'] - df_resultados['R²_CI_Lower'],
              df_resultados['R²_CI_Upper'] - df_resultados['R²']],
        fmt='o-', capsize=5, capthick=2
    )
    axes[0, 0].set_xticks(range(len(df_resultados)))
    axes[0, 0].set_xticklabels(df_resultados['Modelo'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score con Intervalos de Confianza del 95%')
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE con intervalos de confianza
    axes[0, 1].errorbar(
        range(len(df_resultados)), 
        df_resultados['RMSE'],
        yerr=[df_resultados['RMSE'] - df_resultados['RMSE_CI_Lower'],
              df_resultados['RMSE_CI_Upper'] - df_resultados['RMSE']],
        fmt='o-', capsize=5, capthick=2, color='red'
    )
    axes[0, 1].set_xticks(range(len(df_resultados)))
    axes[0, 1].set_xticklabels(df_resultados['Modelo'], rotation=45, ha='right')
    axes[0, 1].set_ylabel('RMSE ($)')
    axes[0, 1].set_title('RMSE con Intervalos de Confianza del 95%')
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE con intervalos de confianza
    axes[1, 0].errorbar(
        range(len(df_resultados)), 
        df_resultados['MAE'],
        yerr=[df_resultados['MAE'] - df_resultados['MAE_CI_Lower'],
              df_resultados['MAE_CI_Upper'] - df_resultados['MAE']],
        fmt='o-', capsize=5, capthick=2, color='green'
    )
    axes[1, 0].set_xticks(range(len(df_resultados)))
    axes[1, 0].set_xticklabels(df_resultados['Modelo'], rotation=45, ha='right')
    axes[1, 0].set_ylabel('MAE ($)')
    axes[1, 0].set_title('MAE con Intervalos de Confianza del 95%')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot con barras de error
    axes[1, 1].errorbar(
        df_resultados['R²'], 
        df_resultados['RMSE'],
        xerr=[df_resultados['R²'] - df_resultados['R²_CI_Lower'],
              df_resultados['R²_CI_Upper'] - df_resultados['R²']],
        yerr=[df_resultados['RMSE'] - df_resultados['RMSE_CI_Lower'],
              df_resultados['RMSE_CI_Upper'] - df_resultados['RMSE']],
        fmt='o', capsize=3, alpha=0.7
    )
    
    # Anotar puntos
    for i, modelo in enumerate(df_resultados['Modelo']):
        axes[1, 1].annotate(modelo, 
                           (df_resultados.iloc[i]['R²'], df_resultados.iloc[i]['RMSE']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_ylabel('RMSE ($)')
    axes[1, 1].set_title('R² vs RMSE con Intervalos de Confianza')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()