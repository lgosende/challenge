
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('optuna').setLevel(logging.WARNING)

# Librerías de modelado
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import optuna
from scipy import stats

# Visualización
import matplotlib
matplotlib.use('Agg')  # Usar backend no-interactivo para evitar problemas con tkinter
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Configurar ruta para funciones utilitarias
src_path = os.path.abspath(os.path.join(os.getcwd(), "../src/"))
if src_path not in sys.path:
    sys.path.append(src_path)

# Configurar matplotlib para evitar problemas con tkinter
def configurar_matplotlib():
    """Configura matplotlib para evitar problemas con GUI"""
    try:
        import matplotlib
        # Usar backend que no requiere GUI
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()  # Desactivar modo interactivo
        return True
    except Exception as e:
        print(f"⚠️ Error configurando matplotlib: {e}")
        return False

# Configurar matplotlib al importar
configurar_matplotlib()

# Importar funciones utilitarias con manejo de errores
try:
    import funciones_utiles as fu
except ImportError:
    print("⚠️ No se pudo importar funciones_utiles. Usando datos sintéticos para pruebas.")
    fu = None

# =============================================================================
# 1. FUNCIONES DE NORMALIZACIÓN
# =============================================================================

class NormalizadorDatos:
    """
    Clase para normalización robusta de datos con diferentes métodos
    """
    
    def __init__(self, metodo='robust'):
        """
        Parámetros:
        -----------
        metodo : str
            'standard', 'robust', 'minmax', 'quantile'
        """
        self.metodo = metodo
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
    def _crear_scaler(self):
        """Crea el scaler según el método especificado"""
        if self.metodo == 'standard':
            return StandardScaler()
        elif self.metodo == 'robust':
            return RobustScaler()  # Mejor para outliers
        elif self.metodo == 'minmax':
            return MinMaxScaler()
        elif self.metodo == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            return QuantileTransformer(output_distribution='normal', n_quantiles=min(100, len(self.X_train)))
        else:
            raise ValueError(f"Método {self.metodo} no soportado")
    
    def fit_transform(self, X_train, feature_names=None):
        """
        Ajusta el normalizador y transforma los datos de entrenamiento
        """
        self.feature_names = feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None)
        self.scaler = self._crear_scaler()
        
        # Convertir a array si es DataFrame
        X_array = X_train.values if hasattr(X_train, 'values') else X_train
        
        X_transformed = self.scaler.fit_transform(X_array)
        self.is_fitted = True
        
        print(f"✅ Normalizador '{self.metodo}' ajustado correctamente")
        print(f"   Forma original: {X_array.shape}")
        print(f"   Rango antes: [{X_array.min():.2f}, {X_array.max():.2f}]")
        print(f"   Rango después: [{X_transformed.min():.2f}, {X_transformed.max():.2f}]")
        
        return X_transformed
    
    def transform(self, X):
        """Transforma nuevos datos usando el normalizador ajustado"""
        if not self.is_fitted:
            raise ValueError("El normalizador debe ser ajustado primero con fit_transform()")
        
        X_array = X.values if hasattr(X, 'values') else X
        return self.scaler.transform(X_array)
    
    def inverse_transform(self, X_scaled):
        """Revierte la normalización"""
        if not self.is_fitted:
            raise ValueError("El normalizador debe ser ajustado primero")
        
        return self.scaler.inverse_transform(X_scaled)
    
    def get_feature_importance_scaled(self, feature_importance, absolute=True):
        """
        Ajusta la importancia de features por la escala de normalización
        """
        if not self.is_fitted:
            return feature_importance
        
        # Para RobustScaler, usar la escala robusta
        if hasattr(self.scaler, 'scale_'):
            importance_scaled = feature_importance / self.scaler.scale_
        else:
            importance_scaled = feature_importance
        
        if absolute:
            importance_scaled = np.abs(importance_scaled)
            
        return importance_scaled

def crear_normalizadores_comparacion():
    """
    Crea múltiples normalizadores para comparar cuál funciona mejor
    """
    normalizadores = {
        'robust': NormalizadorDatos('robust'),
        'standard': NormalizadorDatos('standard'),
        'minmax': NormalizadorDatos('minmax')
    }
    
    return normalizadores

# =============================================================================
# 2. FUNCIONES DE VALIDACIÓN CRUZADA CON INTERVALOS DE CONFIANZA
# =============================================================================

def cross_validation_con_ic(modelo, X, y, cv=5, scoring='r2', nombre_modelo="Modelo"):
    """
    Validación cruzada con intervalos de confianza del 95%
    
    Returns:
    --------
    dict con métricas y intervalos de confianza
    """
    print(f"\n🔄 Ejecutando CV para {nombre_modelo}...")
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores_r2 = []
    scores_rmse = []
    scores_mae = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Dividir datos con manejo consistente de tipos
        if isinstance(X, np.ndarray):
            X_fold_train = X[train_idx]
            X_fold_val = X[val_idx]
        else:
            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            
        if isinstance(y, np.ndarray):
            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]
        else:
            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]
        
        # Entrenar y predecir
        if hasattr(modelo, 'fit'):
            modelo.fit(X_fold_train, y_fold_train)
            y_pred = modelo.predict(X_fold_val)
        else:
            # Para modelos LightGBM con parámetros
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            model_fold = lgb.train(modelo, train_data, num_boost_round=100, callbacks=[lgb.log_evaluation(0)])
            y_pred = model_fold.predict(X_fold_val)
        
        # Calcular métricas
        r2 = r2_score(y_fold_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        mae = mean_absolute_error(y_fold_val, y_pred)
        
        scores_r2.append(r2)
        scores_rmse.append(rmse)
        scores_mae.append(mae)
        if fold == 0:  # Solo mostrar el primer fold
            print(f"   Fold {fold+1}: R²={r2:.4f}, RMSE=${rmse:,.0f}")
    
    # Calcular estadísticas e intervalos de confianza
    def calcular_ic(scores, confidence=0.95):
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        n = len(scores)
        
        # Usar distribución t de Student
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * (std_score / np.sqrt(n))
        
        return {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': mean_score - margin_error,
            'ci_upper': mean_score + margin_error,
            'scores': scores
        }
    
    resultados = {
        'modelo': nombre_modelo,
        'r2': calcular_ic(scores_r2),
        'rmse': calcular_ic(scores_rmse),
        'mae': calcular_ic(scores_mae)
    }
    
    # Imprimir resultados
    print(f"\n📊 Resultados CV para {nombre_modelo}:")
    print(f"   R²: {resultados['r2']['mean']:.4f} ± {resultados['r2']['std']:.4f}")
    print(f"       IC 95%: [{resultados['r2']['ci_lower']:.4f}, {resultados['r2']['ci_upper']:.4f}]")
    print(f"   RMSE: ${resultados['rmse']['mean']:,.0f} ± ${resultados['rmse']['std']:,.0f}")
    print(f"         IC 95%: [${resultados['rmse']['ci_lower']:,.0f}, ${resultados['rmse']['ci_upper']:,.0f}]")
    
    return resultados

# =============================================================================
# 3. OPTIMIZACIÓN CON OPTUNA PARA DATASET PEQUEÑO (CORREGIDA)
# =============================================================================

def objective_lgb_dataset_pequeno(trial, X_train, y_train, categorical_features=None):
    """
    Función objetivo optimizada para LightGBM en datasets pequeños - CORREGIDA
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        #'linear_tree': True,
        
        # Parámetros muy conservadores para dataset pequeño
        'num_leaves': trial.suggest_int('num_leaves', 5, 15),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        
        # Regularización fuerte
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 5.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 80),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 60),
        
        # Submuestreo
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 3),
        
        # Para alta colinealidad
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.1, 1.0),
    }
    
    # Validación cruzada
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_scores = []
    
    try:
        for train_idx, val_idx in kf.split(X_train):
            # Indexación consistente
            if isinstance(X_train, np.ndarray):
                X_fold_train = X_train[train_idx]
                X_fold_val = X_train[val_idx]
            else:
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                
            if isinstance(y_train, np.ndarray):
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
            else:
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
            
            # Crear datasets - SIN categorical_features si causa problemas
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            # Entrenar modelo
            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
            
            # Predecir y evaluar
            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            rmse_scores.append(rmse)
            
    except Exception as e:
        print(f"Error en trial {trial.number}: {e}")
        return float('inf')  # Retornar valor alto si hay error
    
    return np.mean(rmse_scores)

def objective_rf_dataset_pequeno(trial, X_train, y_train):
    """
    Función objetivo optimizada para Random Forest en datasets pequeños - CORREGIDA
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 15, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 8, 25),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Validación cruzada manual
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_scores = []
    
    try:
        for train_idx, val_idx in kf.split(X_train):
            # Indexación consistente
            if isinstance(X_train, np.ndarray):
                X_fold_train = X_train[train_idx]
                X_fold_val = X_train[val_idx]
            else:
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                
            if isinstance(y_train, np.ndarray):
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
            else:
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
            
            # Entrenar modelo
            model = RandomForestRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            
            # Predecir y evaluar
            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            rmse_scores.append(rmse)
            
    except Exception as e:
        print(f"Error en trial {trial.number}: {e}")
        return float('inf')
    
    return np.mean(rmse_scores)

def optimizar_hiperparametros(X_train, y_train, categorical_features=None, n_trials=20, timeout=600):
    """
    Optimiza hiperparámetros para ambos modelos - CORREGIDA
    """
    print("🔧 Iniciando optimización de hiperparámetros...")
    
    # Optimizar LightGBM
    print("\n📊 Optimizando LightGBM...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_lgb = optuna.create_study(direction='minimize', study_name='lgb_small_dataset')
    
    try:
        study_lgb.optimize(
            lambda trial: objective_lgb_dataset_pequeno(trial, X_train, y_train, categorical_features),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,    
            catch=(Exception,)  # Capturar excepciones
        )
        
        print(f"   Mejor RMSE LightGBM: ${study_lgb.best_value:,.0f}")
        print(f"   Mejores parámetros: {study_lgb.best_params}")
        
    except Exception as e:
        print(f"   ⚠️ Error en optimización LightGBM: {e}")
        # Usar parámetros por defecto si falla
        study_lgb.best_params = {
            'num_leaves': 10,
            'max_depth': 4,
            'learning_rate': 0.05,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'min_child_samples': 50,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'min_gain_to_split': 0.5
        }
        study_lgb.best_value = 50000
    
    # Optimizar Random Forest
    print("\n🌲 Optimizando Random Forest...")
    study_rf = optuna.create_study(direction='minimize', study_name='rf_small_dataset')
    
    try:
        study_rf.optimize(
            lambda trial: objective_rf_dataset_pequeno(trial, X_train, y_train),
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,)
        )
        
        print(f"   Mejor RMSE Random Forest: ${study_rf.best_value:,.0f}")
        print(f"   Mejores parámetros: {study_rf.best_params}")
        
    except Exception as e:
        print(f"   ⚠️ Error en optimización Random Forest: {e}")
        # Usar parámetros por defecto si falla
        study_rf.best_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 25,
            'min_samples_leaf': 12,
            'max_features': 'sqrt'
        }
        study_rf.best_value = 50000
    
    return study_lgb, study_rf

# =============================================================================
# 4. FUNCIONES DE VISUALIZACIÓN (CORREGIDAS)
# =============================================================================

def plot_resultados_cv_con_ic(resultados_cv_list, save_path='../output/'):
    """
    Visualiza resultados de validación cruzada con intervalos de confianza
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    modelos = [r['modelo'] for r in resultados_cv_list]
    
    # R² Score
    r2_means = [r['r2']['mean'] for r in resultados_cv_list]
    r2_stds = [r['r2']['std'] for r in resultados_cv_list]
    r2_ci_lower = [r['r2']['ci_lower'] for r in resultados_cv_list]
    r2_ci_upper = [r['r2']['ci_upper'] for r in resultados_cv_list]
    
    x_pos = np.arange(len(modelos))
    axes[0].bar(x_pos, r2_means, yerr=[np.array(r2_means) - np.array(r2_ci_lower),
                                       np.array(r2_ci_upper) - np.array(r2_means)],
                capsize=5, alpha=0.7, color='steelblue')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(modelos, rotation=45, ha='right')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² Score con IC 95%')
    axes[0].grid(True, alpha=0.3)
    
    # RMSE
    rmse_means = [r['rmse']['mean'] for r in resultados_cv_list]
    rmse_ci_lower = [r['rmse']['ci_lower'] for r in resultados_cv_list]
    rmse_ci_upper = [r['rmse']['ci_upper'] for r in resultados_cv_list]
    
    axes[1].bar(x_pos, rmse_means, yerr=[np.array(rmse_means) - np.array(rmse_ci_lower),
                                         np.array(rmse_ci_upper) - np.array(rmse_means)],
                capsize=5, alpha=0.7, color='orange')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(modelos, rotation=45, ha='right')
    axes[1].set_ylabel('RMSE ($)')
    axes[1].set_title('RMSE con IC 95%')
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # MAE
    mae_means = [r['mae']['mean'] for r in resultados_cv_list]
    mae_ci_lower = [r['mae']['ci_lower'] for r in resultados_cv_list]
    mae_ci_upper = [r['mae']['ci_upper'] for r in resultados_cv_list]
    
    axes[2].bar(x_pos, mae_means, yerr=[np.array(mae_means) - np.array(mae_ci_lower),
                                        np.array(mae_ci_upper) - np.array(mae_means)],
                capsize=5, alpha=0.7, color='green')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(modelos, rotation=45, ha='right')
    axes[2].set_ylabel('MAE ($)')
    axes[2].set_title('MAE con IC 95%')
    axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    
    # Guardar figura en lugar de mostrarla
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}cv_resultados_con_ic.png', dpi=300, bbox_inches='tight')
    plt.close()  # Cerrar figura para liberar memoria
    print(f"   ✅ Gráfico guardado: {save_path}cv_resultados_con_ic.png")

def plot_predicciones_vs_reales(y_test, y_pred_lgb, y_pred_rf, nombres_modelos=['LightGBM', 'Random Forest'], save_path='../output/'):
    """
    Visualiza predicciones vs valores reales para ambos modelos
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # LightGBM
    axes[0].scatter(y_test, y_pred_lgb, alpha=0.6, color='blue', s=50)
    min_val, max_val = min(y_test.min(), y_pred_lgb.min()), max(y_test.max(), y_pred_lgb.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Línea perfecta')
    
    # Banda de confianza
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    axes[0].fill_between([min_val, max_val], 
                         [min_val - 1.96*rmse_lgb, max_val - 1.96*rmse_lgb],
                         [min_val + 1.96*rmse_lgb, max_val + 1.96*rmse_lgb],
                         alpha=0.2, color='gray', label=f'IC 95% (±${1.96*rmse_lgb:,.0f})')
    
    axes[0].set_xlabel('Valores Reales ($)')
    axes[0].set_ylabel('Predicciones ($)')
    axes[0].set_title(f'{nombres_modelos[0]} - Predicciones vs Reales')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Random Forest
    axes[1].scatter(y_test, y_pred_rf, alpha=0.6, color='green', s=50)
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Línea perfecta')
    
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    axes[1].fill_between([min_val, max_val], 
                         [min_val - 1.96*rmse_rf, max_val - 1.96*rmse_rf],
                         [min_val + 1.96*rmse_rf, max_val + 1.96*rmse_rf],
                         alpha=0.2, color='gray', label=f'IC 95% (±${1.96*rmse_rf:,.0f})')
    
    axes[1].set_xlabel('Valores Reales ($)')
    axes[1].set_ylabel('Predicciones ($)')
    axes[1].set_title(f'{nombres_modelos[1]} - Predicciones vs Reales')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    
    # Guardar figura
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}predicciones_vs_reales.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico guardado: {save_path}predicciones_vs_reales.png")

def plot_residuos_analisis(y_test, y_pred_lgb, y_pred_rf, save_path='../output/'):
    """
    Análisis detallado de residuos para ambos modelos
    """
    residuos_lgb = y_test - y_pred_lgb
    residuos_rf = y_test - y_pred_rf
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuos vs Predicciones - LightGBM
    axes[0, 0].scatter(y_pred_lgb, residuos_lgb, alpha=0.6, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    std_lgb = np.std(residuos_lgb)
    axes[0, 0].axhline(y=2*std_lgb, color='orange', linestyle=':', alpha=0.7)
    axes[0, 0].axhline(y=-2*std_lgb, color='orange', linestyle=':', alpha=0.7)
    axes[0, 0].set_xlabel('Predicciones ($)')
    axes[0, 0].set_ylabel('Residuos ($)')
    axes[0, 0].set_title('LightGBM - Residuos vs Predicciones')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuos vs Predicciones - Random Forest
    axes[0, 1].scatter(y_pred_rf, residuos_rf, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    std_rf = np.std(residuos_rf)
    axes[0, 1].axhline(y=2*std_rf, color='orange', linestyle=':', alpha=0.7)
    axes[0, 1].axhline(y=-2*std_rf, color='orange', linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel('Predicciones ($)')
    axes[0, 1].set_ylabel('Residuos ($)')
    axes[0, 1].set_title('Random Forest - Residuos vs Predicciones')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribución de residuos - LightGBM
    axes[1, 0].hist(residuos_lgb, bins=15, alpha=0.7, color='blue', density=True)
    mu_lgb, sigma_lgb = np.mean(residuos_lgb), np.std(residuos_lgb)
    x_norm = np.linspace(residuos_lgb.min(), residuos_lgb.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mu_lgb, sigma_lgb)
    axes[1, 0].plot(x_norm, y_norm, 'red', linewidth=2)
    axes[1, 0].set_xlabel('Residuos ($)')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].set_title('LightGBM - Distribución de Residuos')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribución de residuos - Random Forest
    axes[1, 1].hist(residuos_rf, bins=15, alpha=0.7, color='green', density=True)
    mu_rf, sigma_rf = np.mean(residuos_rf), np.std(residuos_rf)
    x_norm = np.linspace(residuos_rf.min(), residuos_rf.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mu_rf, sigma_rf)
    axes[1, 1].plot(x_norm, y_norm, 'red', linewidth=2)
    axes[1, 1].set_xlabel('Residuos ($)')
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].set_title('Random Forest - Distribución de Residuos')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}analisis_residuos.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico guardado: {save_path}analisis_residuos.png")

def plot_feature_importance_comparacion(modelo_lgb, modelo_rf, feature_names, save_path='../output/'):
    """
    Compara la importancia de features entre LightGBM y Random Forest - CORREGIDA
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # LightGBM Feature Importance
    try:
        lgb_importance = modelo_lgb.feature_importance(importance_type='gain')
    except:
        # Si falla, usar split
        lgb_importance = modelo_lgb.feature_importance(importance_type='split')
    
    lgb_df = pd.DataFrame({
        'feature': feature_names[:len(lgb_importance)],  # Asegurar misma longitud
        'importance': lgb_importance
    }).sort_values('importance', ascending=False).head(10)
    
    axes[0].barh(range(len(lgb_df)), lgb_df['importance'])
    axes[0].set_yticks(range(len(lgb_df)))
    axes[0].set_yticklabels(lgb_df['feature'])
    axes[0].set_xlabel('Importancia (Gain)')
    axes[0].set_title('LightGBM - Feature Importance (Top 10)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    
    # Random Forest Feature Importance
    rf_importance = modelo_rf.feature_importances_
    rf_df = pd.DataFrame({
        'feature': feature_names[:len(rf_importance)],  # Asegurar misma longitud
        'importance': rf_importance
    }).sort_values('importance', ascending=False).head(10)
    
    axes[1].barh(range(len(rf_df)), rf_df['importance'])
    axes[1].set_yticks(range(len(rf_df)))
    axes[1].set_yticklabels(rf_df['feature'])
    axes[1].set_xlabel('Importancia')
    axes[1].set_title('Random Forest - Feature Importance (Top 10)')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}feature_importance_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico guardado: {save_path}feature_importance_comparacion.png")
    
    return lgb_df, rf_df


# =============================================================================
# 5. FUNCIONES DE COMPARACIÓN DE MODELOS
# =============================================================================

def crear_tabla_comparativa(resultados_cv_list, y_test, predicciones_dict, baseline_results=None):
    """
    Crea tabla comparativa completa de todos los modelos
    """
    
    # Crear tabla de resultados
    tabla_resultados = []
    
    # Agregar resultados de CV
    for resultado in resultados_cv_list:
        tabla_resultados.append({
            'Modelo': resultado['modelo'],
            'Tipo': 'Avanzado',
            'R²_CV_Mean': resultado['r2']['mean'],
            'R²_CV_Std': resultado['r2']['std'],
            'R²_CV_IC_Lower': resultado['r2']['ci_lower'],
            'R²_CV_IC_Upper': resultado['r2']['ci_upper'],
            'RMSE_CV_Mean': resultado['rmse']['mean'],
            'RMSE_CV_IC_Lower': resultado['rmse']['ci_lower'],
            'RMSE_CV_IC_Upper': resultado['rmse']['ci_upper'],
        })
    
    # Agregar métricas de test
    for nombre_modelo, y_pred in predicciones_dict.items():
        r2_test = r2_score(y_test, y_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_test = mean_absolute_error(y_test, y_pred)
        
        # Buscar el modelo en la tabla y agregar métricas de test
        for i, row in enumerate(tabla_resultados):
            if row['Modelo'] == nombre_modelo:
                tabla_resultados[i].update({
                    'R²_Test': r2_test,
                    'RMSE_Test': rmse_test,
                    'MAE_Test': mae_test
                })
                break
    
    # Agregar resultados baseline si están disponibles
    if baseline_results is not None:
        for _, row in baseline_results.iterrows():
            tabla_resultados.append({
                'Modelo': row['Modelo'],
                'Tipo': 'Baseline',
                'R²_Test': row['R²'],
                'RMSE_Test': row['RMSE'],
                'MAE_Test': row['MAE'],
                'R²_CV_Mean': np.nan,
                'R²_CV_Std': np.nan,
                'R²_CV_IC_Lower': np.nan,
                'R²_CV_IC_Upper': np.nan,
                'RMSE_CV_Mean': np.nan,
                'RMSE_CV_IC_Lower': np.nan,
                'RMSE_CV_IC_Upper': np.nan,
            })
    
    df_comparacion = pd.DataFrame(tabla_resultados)
    
    # Ordenar por R² de test (descendente) - NaNs al final por defecto
    df_comparacion = df_comparacion.sort_values('R²_Test', ascending=False, na_position='last')
    
    return df_comparacion

def generar_reporte_completo(df_comparacion, save_path='../output/'):
    """
    Genera reporte completo con análisis de modelos
    """
    
    print("=" * 80)
    print("                    REPORTE COMPARATIVO DE MODELOS")
    print("=" * 80)
    
    # Mejor modelo
    if not df_comparacion.empty:
        mejor_modelo = df_comparacion.iloc[0]
        print(f"\n🏆 MEJOR MODELO: {mejor_modelo['Modelo']}")
        print(f"   Tipo: {mejor_modelo['Tipo']}")
        print(f"   R² Test: {mejor_modelo['R²_Test']:.4f}")
        print(f"   RMSE Test: ${mejor_modelo['RMSE_Test']:,.0f}")
        
        if not pd.isna(mejor_modelo['R²_CV_Mean']):
            print(f"   R² CV: {mejor_modelo['R²_CV_Mean']:.4f} ± {mejor_modelo['R²_CV_Std']:.4f}")
            print(f"   R² CV IC 95%: [{mejor_modelo['R²_CV_IC_Lower']:.4f}, {mejor_modelo['R²_CV_IC_Upper']:.4f}]")
    
    # Comparación por tipo de modelo
    print(f"\n📊 MODELOS POR TIPO:")
    tipos = df_comparacion['Tipo'].value_counts()
    for tipo, count in tipos.items():
        print(f"   {tipo}: {count} modelos")
        modelos_tipo = df_comparacion[df_comparacion['Tipo'] == tipo]
        mejor_tipo = modelos_tipo.iloc[0] if not modelos_tipo.empty else None
        if mejor_tipo is not None:
            print(f"      Mejor: {mejor_tipo['Modelo']} (R² = {mejor_tipo['R²_Test']:.4f})")
    
    # Análisis de mejora
    baseline_models = df_comparacion[df_comparacion['Tipo'] == 'Baseline']
    advanced_models = df_comparacion[df_comparacion['Tipo'] == 'Avanzado']
    
    if not baseline_models.empty and not advanced_models.empty:
        mejor_baseline = baseline_models.iloc[0]
        mejor_avanzado = advanced_models.iloc[0]
        
        mejora_r2 = mejor_avanzado['R²_Test'] - mejor_baseline['R²_Test']
        mejora_rmse = mejor_baseline['RMSE_Test'] - mejor_avanzado['RMSE_Test']
        mejora_r2_pct = (mejora_r2 / mejor_baseline['R²_Test']) * 100
        mejora_rmse_pct = (mejora_rmse / mejor_baseline['RMSE_Test']) * 100
        
        print(f"\n📈 MEJORA DE MODELOS AVANZADOS:")
        print(f"   Mejor Baseline: {mejor_baseline['Modelo']} (R² = {mejor_baseline['R²_Test']:.4f})")
        print(f"   Mejor Avanzado: {mejor_avanzado['Modelo']} (R² = {mejor_avanzado['R²_Test']:.4f})")
        print(f"   Mejora R²: {mejora_r2:+.4f} ({mejora_r2_pct:+.1f}%)")
        print(f"   Mejora RMSE: ${mejora_rmse:+,.0f} ({mejora_rmse_pct:+.1f}%)")
        
        if mejora_r2 > 0.01:
            print("   ✅ Mejora SIGNIFICATIVA")
        elif mejora_r2 > 0.002:
            print("   ⚠️  Mejora MARGINAL")
        else:
            print("   ❌ Sin mejora significativa")
    
    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)
    
    # Guardar resultados
    df_comparacion.to_csv(f'{save_path}comparacion_modelos_completa.csv', index=False)
    print(f"\n💾 Tabla comparativa guardada en: {save_path}comparacion_modelos_completa.csv")
    
    print("=" * 80)
    
    return df_comparacion

# =============================================================================
# 6. FUNCIÓN PRINCIPAL DEL PIPELINE (CORREGIDA)
# =============================================================================

def crear_datos_sinteticos():
    """
    Crea datos sintéticos para pruebas cuando no están disponibles los reales
    """
    np.random.seed(42)
    n_samples = 370
    
    # Variables base
    age = np.random.normal(35, 8, n_samples)
    years_experience = np.random.exponential(7, n_samples)
    education_numeric = np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.3, 0.4, 0.2])
    
    # Variables derivadas
    ratio_experiencia = years_experience / age
    experience2 = years_experience ** 2
    edad_x_experiencia = age * years_experience
    log_experiencia = np.log1p(years_experience)
    
    # Variables categóricas
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4])
    education_level = np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples, p=[0.5, 0.3, 0.2])
    seniority = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.3, 0.25, 0.15])
    functional_area = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])
    
    # Variable objetivo (salary) con ruido
    salary_base = (
        50000 +
        years_experience * 3000 +
        education_numeric * 15000 +
        age * 500 +
        seniority * 10000 +
        functional_area * 5000
    )
    salary = salary_base + np.random.normal(0, 8000, n_samples)
    salary = np.clip(salary, 30000, 200000)  # Limitar rango realista
    
    # Crear DataFrame
    df = pd.DataFrame({
        'age': age,
        'years_experience': years_experience,
        'education_numerica': education_numeric,
        'ratio_experiencia': ratio_experiencia,
        'experience2': experience2,
        'edad_x_experiencia': edad_x_experiencia,
        'log_experiencia': log_experiencia,
        'gender': gender,
        'education_level': education_level,
        'llm_seniority': seniority,
        'llm_functional_area': functional_area,
        'salary': salary,
        
        # Variables adicionales
        'inicio_laboral': age - years_experience,
        'ratio_velocidad_carrera': years_experience / (age - 18),
        'experience3': years_experience ** 3,
        'edad_x_experiencia2': age * (years_experience ** 2),
        'educacion_x_experiencia': education_numeric * years_experience,
        'gender_education': gender + '_' + education_level
    })
    
    return df

def preparar_datos():
    """
    Carga y prepara los datos para modelado - CORREGIDA
    """
    print("📊 Cargando y preparando datos...")
    
    # Intentar cargar dataset real
    try:
        if fu is not None:
            df = fu.ejecutar_query("SELECT * FROM df_feature_engineering")
            print(f"✅ Datos reales cargados: {df.shape}")
        else:
            raise ImportError("funciones_utiles no disponible")
    except:
        print("⚠️ No se pudieron cargar datos reales. Usando datos sintéticos...")
        df = crear_datos_sinteticos()
        print(f"✅ Datos sintéticos creados: {df.shape}")
    
    # Definir variables
    variables_onehot = ['gender', 'education_level', 'gender_education']
    variables_categoricas = ['llm_seniority', 'llm_functional_area']
    variables_numericas = [
        'age', 'years_experience', 'inicio_laboral', 'ratio_experiencia',
        'ratio_velocidad_carrera', 'experience2', 'experience3',
        'edad_x_experiencia', 'edad_x_experiencia2', 'log_experiencia', 
        'education_numerica', 'educacion_x_experiencia'
    ]
    
    # Filtrar variables existentes
    variables_onehot = [var for var in variables_onehot if var in df.columns]
    variables_categoricas = [var for var in variables_categoricas if var in df.columns]
    variables_numericas = [var for var in variables_numericas if var in df.columns]
    
    target = 'salary'
    
    # Limpiar datos
    variables_todas = variables_onehot + variables_categoricas + variables_numericas + [target]
    df_clean = df.dropna(subset=variables_todas)
    
    print(f"Registros después de limpieza: {len(df_clean)}")
    
    # Preparar X y y
    X_num = df_clean[variables_numericas].copy()
    
    # One-hot encoding
    for var in variables_onehot:
        if var in df_clean.columns:
            dummies = pd.get_dummies(df_clean[var], prefix=var, drop_first=True)
            X_num = pd.concat([X_num, dummies], axis=1)
    
    # Label encoding para variables categóricas
    categorical_features = []
    for var in variables_categoricas:
        if var in df_clean.columns:
            le = LabelEncoder()
            X_num[f'{var}_encoded'] = le.fit_transform(df_clean[var].astype(str))
            categorical_features.append(f'{var}_encoded')
    
    y = df_clean[target].copy()
    
    return X_num, y, categorical_features, df_clean

def ejecutar_pipeline_completo(n_trials=15, timeout=900, test_size=0.2, cv_folds=5):
    """
    Ejecuta el pipeline completo de modelado avanzado - CORREGIDA
    """
    
    print("🚀 INICIANDO PIPELINE DE MODELADO AVANZADO")
    print("=" * 60)
    
    try:
        # 1. Preparar datos
        X, y, categorical_features, df_clean = preparar_datos()
        
        # 2. División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Dataset dividido: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        # 3. Normalización de datos
        print("\n🔧 Normalizando datos...")
        normalizador = NormalizadorDatos('robust')
        X_train_scaled = normalizador.fit_transform(X_train, X_train.columns.tolist())
        X_test_scaled = normalizador.transform(X_test)
        
        # 4. Optimización de hiperparámetros
        print("\n🔍 Optimizando hiperparámetros...")
        study_lgb, study_rf = optimizar_hiperparametros(
            X_train_scaled, y_train, categorical_features, n_trials, timeout
        )
        
        # 5. Entrenar modelos finales
        print("\n🤖 Entrenando modelos finales...")
        
        # LightGBM final
        lgb_params_final = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
        }
        lgb_params_final.update(study_lgb.best_params)
        
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        modelo_lgb_final = lgb.train(
            lgb_params_final,
            train_data,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Random Forest final
        rf_params_final = study_rf.best_params.copy()
        rf_params_final.update({'random_state': 42, 'n_jobs': -1})
        modelo_rf_final = RandomForestRegressor(**rf_params_final)
        modelo_rf_final.fit(X_train_scaled, y_train)
        
        # 6. Predicciones
        print("\n📊 Generando predicciones...")
        y_pred_lgb = modelo_lgb_final.predict(X_test_scaled)
        y_pred_rf = modelo_rf_final.predict(X_test_scaled)
        
        predicciones_dict = {
            'LightGBM': y_pred_lgb,
            'Random Forest': y_pred_rf
        }
        
        # 7. Validación cruzada con intervalos de confianza
        print("\n🔄 Ejecutando validación cruzada...")
        
        # LGBWrapper corregido
        class LGBWrapper:
            def __init__(self, params):
                self.params = params.copy() if isinstance(params, dict) else params
                self.model = None
                
            def get_params(self, deep=True):
                """Implementar get_params para compatibilidad con sklearn"""
                return {'params': self.params}
                
            def set_params(self, **parameters):
                """Implementar set_params para compatibilidad con sklearn"""
                if 'params' in parameters:
                    self.params = parameters['params']
                return self
                
            def fit(self, X, y):
                train_data = lgb.Dataset(X, label=y)
                self.model = lgb.train(
                    self.params, 
                    train_data, 
                    num_boost_round=200, 
                    callbacks=[lgb.log_evaluation(0)]
                )
                return self
                
            def predict(self, X):
                if self.model is None:
                    raise ValueError("Modelo no ha sido entrenado")
                return self.model.predict(X)
        
        lgb_wrapper = LGBWrapper(lgb_params_final)
        rf_wrapper = RandomForestRegressor(**rf_params_final)
        
        resultados_cv = [
            cross_validation_con_ic(lgb_wrapper, X_train_scaled, y_train, cv_folds, nombre_modelo="LightGBM"),
            cross_validation_con_ic(rf_wrapper, X_train_scaled, y_train, cv_folds, nombre_modelo="Random Forest")
        ]
        
        # 8. Cargar resultados baseline si existen
        baseline_results = None
        try:
            baseline_results = pd.read_csv('../data/resultados_baseline_models.csv')
            print("✅ Resultados baseline cargados")
        except:
            print("⚠️  No se encontraron resultados baseline")
        
        # 9. Crear tabla comparativa
        print("\n📋 Creando tabla comparativa...")
        df_comparacion = crear_tabla_comparativa(
            resultados_cv, y_test, predicciones_dict, baseline_results
        )
        
        # 10. Generar visualizaciones
        print("\n📈 Generando visualizaciones...")
        
        print("   Gráfico 1: Resultados CV con IC...")
        plot_resultados_cv_con_ic(resultados_cv)
        
        print("   Gráfico 2: Predicciones vs Reales...")
        plot_predicciones_vs_reales(y_test, y_pred_lgb, y_pred_rf)
        
        print("   Gráfico 3: Análisis de residuos...")
        plot_residuos_analisis(y_test, y_pred_lgb, y_pred_rf)
        
        print("   Gráfico 4: Feature importance...")
        lgb_importance_df, rf_importance_df = plot_feature_importance_comparacion(
            modelo_lgb_final, modelo_rf_final, X_train.columns.tolist()
        )
        
        # 11. Generar reporte completo
        print("\n📄 Generando reporte completo...")
        df_comparacion_final = generar_reporte_completo(df_comparacion)
        
        # 12. Calcular intervalos de confianza para predicciones
        print("\n🎯 Calculando intervalos de confianza para predicciones...")
        
        def calcular_ic_predicciones_bootstrap(modelo, X_train, y_train, X_test, n_bootstrap=100):
            """Calcula IC para predicciones usando bootstrap - versión simplificada"""
            predicciones_bootstrap = []
            
            for i in range(n_bootstrap):
                try:
                    # Muestreo bootstrap
                    n_samples = len(X_train)
                    bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
                    X_boot = X_train[bootstrap_idx]
                    y_boot = y_train.iloc[bootstrap_idx] if hasattr(y_train, 'iloc') else y_train[bootstrap_idx]
                    
                    # Entrenar modelo según el tipo
                    if isinstance(modelo, LGBWrapper):
                        # Para LightGBM wrapper, crear nuevo modelo con mismos parámetros
                        modelo_boot = LGBWrapper(modelo.params.copy())
                        modelo_boot.fit(X_boot, y_boot)
                        pred_boot = modelo_boot.predict(X_test)
                    elif hasattr(modelo, 'get_params'):
                        # Para modelos sklearn (Random Forest)
                        from sklearn.base import clone
                        modelo_boot = clone(modelo)
                        modelo_boot.fit(X_boot, y_boot)
                        pred_boot = modelo_boot.predict(X_test)
                    else:
                        # Fallback: crear nuevo modelo del mismo tipo
                        modelo_boot = type(modelo)()
                        modelo_boot.fit(X_boot, y_boot)
                        pred_boot = modelo_boot.predict(X_test)
                    
                    predicciones_bootstrap.append(pred_boot)
                    
                except Exception as e:
                    if i < 5:  # Solo mostrar primeros errores
                        print(f"   Error en bootstrap {i}: {e}")
                    continue
            
            if len(predicciones_bootstrap) == 0:
                # Si no hay predicciones válidas, usar predicción original con estimación conservadora
                if isinstance(modelo, LGBWrapper):
                    pred_original = modelo.predict(X_test)
                else:
                    pred_original = modelo.predict(X_test)
                    
                pred_std = np.std(pred_original) * 0.1  # Estimación conservadora del 10%
                return pred_original, pred_original - 1.96*pred_std, pred_original + 1.96*pred_std
            
            predicciones_bootstrap = np.array(predicciones_bootstrap)
            
            # Calcular intervalos
            pred_mean = np.mean(predicciones_bootstrap, axis=0)
            pred_lower = np.percentile(predicciones_bootstrap, 2.5, axis=0)
            pred_upper = np.percentile(predicciones_bootstrap, 97.5, axis=0)
            
            return pred_mean, pred_lower, pred_upper
        
        # IC para LightGBM
        lgb_pred_mean, lgb_pred_lower, lgb_pred_upper = calcular_ic_predicciones_bootstrap(
            lgb_wrapper, X_train_scaled, y_train, X_test_scaled, n_bootstrap=50
        )
        
        # IC para Random Forest  
        rf_pred_mean, rf_pred_lower, rf_pred_upper = calcular_ic_predicciones_bootstrap(
            modelo_rf_final, X_train_scaled, y_train, X_test_scaled, n_bootstrap=50
        )
        
        print("✅ Intervalos de confianza calculados")
        
        # 13. Compilar resultados finales
        resultados_finales = {
            'datos': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'feature_names': X_train.columns.tolist()
            },
            'modelos': {
                'lgb': modelo_lgb_final,
                'rf': modelo_rf_final,
                'normalizador': normalizador
            },
            'predicciones': {
                'lgb': {
                    'pred': y_pred_lgb,
                    'pred_mean': lgb_pred_mean,
                    'pred_lower': lgb_pred_lower,
                    'pred_upper': lgb_pred_upper
                },
                'rf': {
                    'pred': y_pred_rf,
                    'pred_mean': rf_pred_mean,
                    'pred_lower': rf_pred_lower,
                    'pred_upper': rf_pred_upper
                }
            },
            'validacion_cruzada': resultados_cv,
            'comparacion': df_comparacion_final,
            'feature_importance': {
                'lgb': lgb_importance_df,
                'rf': rf_importance_df
            },
            'optimizacion': {
                'lgb_study': study_lgb,
                'rf_study': study_rf
            }
        }
        
        print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        return resultados_finales
        
    except Exception as e:
        print(f"\n❌ ERROR EN PIPELINE: {e}")
        print("Traceback completo:")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# 7. FUNCIONES DE UTILIDAD PARA USO EN NOTEBOOK
# =============================================================================

def predecir_nuevos_datos(modelo, normalizador, X_nuevos, tipo_modelo='rf'):
    """
    Predice nuevos datos usando el modelo entrenado
    """
    try:
        # Normalizar datos
        X_nuevos_scaled = normalizador.transform(X_nuevos)
        
        # Predecir
        if tipo_modelo == 'lgb':
            if hasattr(modelo, 'predict'):
                predicciones = modelo.predict(X_nuevos_scaled)
            else:
                predicciones = modelo.model.predict(X_nuevos_scaled)
        else:
            predicciones = modelo.predict(X_nuevos_scaled)
        
        return predicciones
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return None

def mostrar_resumen_resultados(resultados):
    """
    Muestra un resumen de los resultados del pipeline
    """
    if resultados is None:
        print("❌ No hay resultados para mostrar")
        return
        
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 50)
    
    # Mejor modelo
    mejor_modelo = resultados['comparacion'].iloc[0]
    print(f"🏆 Mejor modelo: {mejor_modelo['Modelo']}")
    print(f"   R² Test: {mejor_modelo['R²_Test']:.4f}")
    print(f"   RMSE Test: ${mejor_modelo['RMSE_Test']:,.0f}")
    
    # Estadísticas de predicciones
    y_test = resultados['datos']['y_test']
    
    for modelo_name in ['lgb', 'rf']:
        pred = resultados['predicciones'][modelo_name]['pred']
        pred_lower = resultados['predicciones'][modelo_name]['pred_lower']
        pred_upper = resultados['predicciones'][modelo_name]['pred_upper']
        
        ancho_ic_promedio = np.mean(pred_upper - pred_lower)
        
        print(f"\n📈 {modelo_name.upper()}:")
        print(f"   R²: {r2_score(y_test, pred):.4f}")
        print(f"   RMSE: ${np.sqrt(mean_squared_error(y_test, pred)):,.0f}")
        print(f"   Ancho IC promedio: ${ancho_ic_promedio:,.0f}")
    
    print("=" * 50)

# =============================================================================
# 8. INSTRUCCIONES DE USO DESDE NOTEBOOK
# =============================================================================

def instrucciones_uso():
    """
    Muestra las instrucciones de uso desde un notebook
    """
    print("""
    ═══════════════════════════════════════════════════════════════
                    INSTRUCCIONES DE USO DESDE NOTEBOOK
    ═══════════════════════════════════════════════════════════════
    
    1. CONFIGURACIÓN INICIAL:
    
    ```python
    import sys
    sys.path.append('../src/')
    import modelos_avanzados as ma
    ```
    
    2. EJECUTAR PIPELINE COMPLETO:
    
    ```python
    # Ejecutar todo el análisis
    resultados = ma.ejecutar_pipeline_completo(
        n_trials=20,      # Número de trials de Optuna
        timeout=900,      # Timeout en segundos (15 min)
        test_size=0.2,    # 20% para test
        cv_folds=5        # 5 folds para CV
    )
    ```
    
    3. VER RESUMEN DE RESULTADOS:
    
    ```python
    ma.mostrar_resumen_resultados(resultados)
    ```
    
    4. ACCEDER A COMPONENTES ESPECÍFICOS:
    
    ```python
    # Modelos entrenados
    modelo_lgb = resultados['modelos']['lgb']
    modelo_rf = resultados['modelos']['rf']
    normalizador = resultados['modelos']['normalizador']
    
    # Tabla comparativa
    tabla_comparacion = resultados['comparacion']
    print(tabla_comparacion)
    
    # Predicciones con IC
    predicciones_lgb = resultados['predicciones']['lgb']
    predicciones_rf = resultados['predicciones']['rf']
    ```
    
    5. PREDECIR NUEVOS DATOS:
    
    ```python
    # Preparar nuevos datos (mismo formato que entrenamiento)
    nuevos_datos = ...  # DataFrame con las mismas columnas
    
    # Predecir con LightGBM
    pred_lgb = ma.predecir_nuevos_datos(
        modelo_lgb, normalizador, nuevos_datos, 'lgb'
    )
    
    # Predecir con Random Forest
    pred_rf = ma.predecir_nuevos_datos(
        modelo_rf, normalizador, nuevos_datos, 'rf'
    )
    ```
    
    6. ARCHIVOS GENERADOS:
    
    - ../output/comparacion_modelos_completa.csv
    - Gráficos de validación cruzada
    - Gráficos de predicciones vs reales
    - Análisis de residuos
    - Feature importance comparativo
    
    7. SOLUCIÓN DE PROBLEMAS:
    
    - Si funciones_utiles no está disponible, el script usará datos sintéticos
    - Si hay errores en Optuna, se usarán parámetros por defecto
    - Todos los errores son capturados y el pipeline continúa
    
    8. EJEMPLO COMPLETO:
    
    ```python
    import sys
    sys.path.append('../src/')
    import modelos_avanzados as ma
    
    # Ejecutar pipeline
    resultados = ma.ejecutar_pipeline_completo(n_trials=10, timeout=300)
    
    # Ver resumen
    if resultados is not None:
        ma.mostrar_resumen_resultados(resultados)
        
        # Acceder a mejor modelo
        mejor_modelo_row = resultados['comparacion'].iloc[0]
        mejor_modelo_nombre = mejor_modelo_row['Modelo']
        
        if mejor_modelo_nombre == 'LightGBM':
            mejor_modelo = resultados['modelos']['lgb']
        else:
            mejor_modelo = resultados['modelos']['rf']
        
        print(f"Mejor modelo: {mejor_modelo_nombre}")
    else:
        print("Hubo errores en el pipeline")
    ```
    
    ═══════════════════════════════════════════════════════════════
    """)

# =============================================================================
# 9. FUNCIÓN PARA PRUEBAS RÁPIDAS
# =============================================================================

def ejecutar_prueba_rapida():
    """
    Ejecuta una prueba rápida del pipeline con parámetros reducidos
    """
    print("🧪 EJECUTANDO PRUEBA RÁPIDA DEL PIPELINE")
    print("=" * 50)
    
    try:
        resultados = ejecutar_pipeline_completo(
            n_trials=5,      # Pocos trials para velocidad
            timeout=120,     # 2 minutos máximo
            test_size=0.3,   # Más datos para test
            cv_folds=3       # Menos folds
        )
        
        if resultados is not None:
            print("\n✅ PRUEBA EXITOSA")
            mostrar_resumen_resultados(resultados)
            return True
        else:
            print("\n❌ PRUEBA FALLÓ")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBA: {e}")
        return False

# =============================================================================
# 10. MAIN Y PUNTO DE ENTRADA
# =============================================================================

def main():
    """
    Función principal que se ejecuta al correr el script directamente
    """
    print("🚀 MODELOS AVANZADOS - SCRIPT DE ANÁLISIS")
    print("=" * 60)
    
    print("\nOpciones disponibles:")
    print("1. Ejecutar pipeline completo")
    print("2. Ejecutar prueba rápida")
    print("3. Mostrar instrucciones")
    print("4. Salir")
    
    while True:
        try:
            opcion = input("\nSeleccione una opción (1-4): ").strip()
            
            if opcion == '1':
                print("\n🔧 Configurando pipeline completo...")
                
                # Parámetros personalizables
                try:
                    n_trials = int(input("Número de trials de Optuna (default 15): ") or "15")
                    timeout = int(input("Timeout en segundos (default 900): ") or "900")
                    cv_folds = int(input("Número de folds CV (default 5): ") or "5")
                except ValueError:
                    print("Usando valores por defecto...")
                    n_trials, timeout, cv_folds = 15, 900, 5
                
                resultados = ejecutar_pipeline_completo(
                    n_trials=n_trials,
                    timeout=timeout,
                    cv_folds=cv_folds
                )
                
                if resultados is not None:
                    mostrar_resumen_resultados(resultados)
                break
                
            elif opcion == '2':
                ejecutar_prueba_rapida()
                break
                
            elif opcion == '3':
                instrucciones_uso()
                
            elif opcion == '4':
                print("👋 ¡Hasta luego!")
                break
                
            else:
                print("❌ Opción no válida. Por favor seleccione 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            break

if __name__ == "__main__":
    # Si se ejecuta directamente, mostrar menú
    main()
else:
    # Si se importa como módulo, mostrar mensaje informativo
    print("📦 Módulo modelos_avanzados cargado correctamente")
    print("💡 Tip: Ejecuta ma.instrucciones_uso() para ver las instrucciones")
    print("🧪 Tip: Ejecuta ma.ejecutar_prueba_rapida() para una prueba rápida")