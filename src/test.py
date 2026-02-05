# Importaciones 
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import numpy as np
import time
from codecarbon import OfflineEmissionsTracker
from sklearn.preprocessing import MinMaxScaler
from sklearn import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor,
                              HistGradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from autogluon.common import space as ag
from skopt import BayesSearchCV

from skopt.space import Real, Integer, Categorical
import os, hashlib, random, torch

SEED = 42

def seed_from_pair(compañia: str, modelo: str, base_seed: int = 42) -> int:
    h = hashlib.sha256(f"{compañia}-{modelo}-{base_seed}".encode()).hexdigest()
    return int(h[:8], 16)

def set_all_seeds(seed:int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


df = pd.read_csv("DatosFinanzasFFill.csv")

df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# Los modelos de autogluon necesitan un id si es un problema de multiserie, añadimos artificialmente un 0
df["id"]=0


def evaluar_modelo_autogluon(nombre_dataset, df, frecuencia, modelo, seed,
                          initial_train_size=0.8, prediction_length=1, context_len=30, evalM="sMAPE",
                          item_id_col="item_id", target="target", date="date"):
    
    df = df.dropna(subset=[target])
    df = df.dropna(axis=1)           # eliminamos columnas que sigan teniendo NaN
    # Si no hay columna de id (no hay más de una serie temporal), se crea artificialmente
    if item_id_col is None: 
        df["item_id"] = 0
    else:
        df = df.rename(columns={item_id_col:"item_id"})
    # Se pasa a formato datetime la columna de fecha
    df["date"] = pd.to_datetime(df[date])
    df2 = df

    full_df = df2[df2["item_id"] == 0]
    full_series = full_df[target].values #serie a predecir 
    total_len = len(full_series)
    
    past_covariates_cols = [
        col for col in df.columns
        if col not in {target, "item_id", date, "date"}
    ]

    initial_len = int(initial_train_size * len(df2))
    train_df = df2.iloc[:initial_len]
    test_df  = df2.iloc[initial_len:]

    
    # Escalamos con MinMax el conjunto de entrenamiento y aplicamos la transformación al resto de la serie 
    columns_to_scale = past_covariates_cols+[target] 
    scaler = MinMaxScaler().fit(train_df[columns_to_scale])
    train_df[columns_to_scale] = scaler.transform(train_df[columns_to_scale])
    full_df[columns_to_scale] = scaler.transform(full_df[columns_to_scale])

    params = pd.DataFrame({
        'min_dato': scaler.data_min_,
        'max_dato': scaler.data_max_,
        'rango': scaler.data_range_,
        'offset_usado': scaler.min_,
        'factor_escala': scaler.scale_
    }, index=columns_to_scale)

    min_target = params.loc[target, 'min_dato']
    max_target = params.loc[target, 'max_dato']

    train_df = train_df.rename(columns={target:"target"})
    full_df = full_df.rename(columns={target:"target"})
    ts_train = TimeSeriesDataFrame.from_data_frame(train_df, timestamp_column="date")

    # Configurar hiperparámetros por modelo -> Tenemos busqueda de los mejores hiperparámetros

    if modelo == "DeepAR":
        hyperparams = {
            "DeepAR": {
                "use_past_covariates": True,
                "max_epochs": 40,
                "lr": ag.Real(1e-4, 3e-3, log=True),
                "dropout": ag.Real(0.0, 0.3),
                "hidden_size": ag.Int(32, 256),
                "num_layers": ag.Int(1, 4),
                "context_length": context_len,
            }
        }

    elif modelo == "PatchTST":
        hyperparams = {
            "PatchTST": {
                "max_epochs": 40,
                "lr": ag.Real(1e-4, 5e-3, log=True),
                "dropout": ag.Real(0.0, 0.3),
                "d_model": ag.Categorical(64, 128, 256),
                "nhead": ag.Categorical(4, 8),
                "patch_len": ag.Categorical(8, 16, 24, 32),
                "stride": ag.Categorical(4, 8, 16),
                "context_length": context_len,
            }
        }
    
    elif modelo == "TemporalFusionTransformer":
        hyperparams = {
            "TemporalFusionTransformer": {
                "max_epochs": 40,
                "lr": ag.Real(1e-4, 5e-3, log=True),
                "dropout": ag.Real(0.0, 0.3),
                "hidden_size": ag.Categorical(64, 128),
                "num_layers": ag.Int(1, 3),
                "batch_size": ag.Categorical(32, 64, 128),
                "context_length": context_len,
            }
        }
    
    else:
        hyperparams = {modelo: {}}


    # Entrenamiento único
    print(f"\n📊 === Entrenando modelo '{modelo}' una sola vez con {initial_len} observaciones ===")
    #set_all_seeds(seed)
    tracker = OfflineEmissionsTracker(
        project_name="Chronos_experiments",
        measure_power_secs=10,
        country_iso_code="ESP",
        tracking_mode="process",
        log_level="error",
        save_to_file=False,
    )
    tracker.start()
    start_train_time = time.time()
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric= evalM,
        freq=frecuencia,
        verbosity=4
        ).fit(ts_train, random_seed=seed,hyperparameters=hyperparams,     
              hyperparameter_tune_kwargs={
                "num_trials": 10,
                "scheduler": "local",
                "searcher": "bayes",
            }
)
    training_time = time.time() - start_train_time # Forma algo "artesanal" de tomar el tiempo de entrenamiento 
    train_emissions = tracker.stop()


    pred_list = []
    real_list = []


    print(f"\n📊 === Backtesting con modelo fijo: {nombre_dataset} ===")

    inference_time = 0
    inference_emissions_count= 0 
    for start in range(initial_len, total_len - prediction_length + 1, prediction_length):
        #set_all_seeds(seed)
        # end -> donde termina la predicción 
        end = start + prediction_length
        # start_context -> marca el inicio del contexto  
        start_context = max(0, start - context_len)
        # context_series -> contexto con el que predecir
        context_series = full_df.iloc[start_context:start] 
        ts_context = TimeSeriesDataFrame.from_data_frame(context_series, timestamp_column="date")

        # Realizamos la predicción con el modelo ya entrenado (predictor) y almacenamos el tiempo y carbono
        tracker = OfflineEmissionsTracker(
            project_name="Chronos_experiments",
            measure_power_secs=10,
            country_iso_code="ESP",
            tracking_mode="process",
            log_level="error",
            save_to_file=False
        )
        tracker.start()
        start_pred_time = time.time()
        pred = predictor.predict(ts_context, random_seed=seed)
        tiempo = time.time() - start_pred_time
        inference_emissions = tracker.stop()
        inference_time = inference_time+tiempo 
        inference_emissions_count= inference_emissions_count+inference_emissions
        # Tomamos la media como valor puntual predicho, tomamos los valores reales y calculamos la bondad de ajuste 
        pred_val = pred["mean"].iloc[0]
        true_val = full_series[start:end]

        rango_target = max_target - min_target
        pred_val_desescalado = pred_val * rango_target + min_target

        pred_list.append(pred_val_desescalado)
        real_list.append(true_val[0])

    pred_col_name = f"pred_{modelo}"

    pred_vs_real_df = pd.DataFrame({
        "real": real_list,
        pred_col_name: pred_list
    })

    return pred_vs_real_df, training_time, train_emissions, inference_time, inference_emissions_count

from chronos.chronos2.pipeline import Chronos2Pipeline
from chronos.base import BaseChronosPipeline
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

def evaluar_modelo_chronos2(nombre_dataset, df, frecuencia, seed,
                          initial_train_size=0.8, prediction_length=1, context_len=30, evalM="SMAPE",
                          item_id_col="item_id", target="target", date="date"):
    
    df = df.dropna(subset=[target])
    df = df.dropna(axis=1)           # eliminamos columnas que sigan teniendo NaN
    # Si no hay columna de id (no hay más de una serie temporal), se crea artificialmente
    if item_id_col is None: 
        df["item_id"] = 0
    else:
        df = df.rename(columns={item_id_col:"item_id"})
    
    # Se pasa a formato datetime la columna de fecha
    df[date] = pd.to_datetime(df[date])
    df2 = df

    full_df = df2[df2["item_id"] == 0]
    full_series = full_df[target].values #serie a predecir 
    total_len = len(full_series)
    
    past_covariates_cols = [
        col for col in df.columns
        if col not in {target, "item_id", date, "date"}
    ]

    initial_len = int(initial_train_size * len(df2))
    train_df = df2.iloc[:initial_len]
    test_df  = df2.iloc[initial_len:]

    
    # Escalamos con MinMax el conjunto de entrenamiento y aplicamos la transformación al resto de la serie 
    columns_to_scale = past_covariates_cols+[target] 
    scaler = MinMaxScaler().fit(train_df[columns_to_scale])
    train_df[columns_to_scale] = scaler.transform(train_df[columns_to_scale])
    full_df[columns_to_scale] = scaler.transform(full_df[columns_to_scale])

    params = pd.DataFrame({
        'min_dato': scaler.data_min_,
        'max_dato': scaler.data_max_,
        'rango': scaler.data_range_,
        'offset_usado': scaler.min_,
        'factor_escala': scaler.scale_
    }, index=columns_to_scale)

    min_target = params.loc[target, 'min_dato']
    max_target = params.loc[target, 'max_dato']

    full_df = full_df.rename(columns={target:"target"})

    # Lista de métricas -> las almacenamos para cada bloque y las promediamos posteriormente 
    pred_list = []
    real_list = []

    print(f"\n📊 === Backtesting con modelo fijo: {nombre_dataset} ===")

    inference_time = 0
    inference_emissions_count= 0 
    for start in range(initial_len, total_len - prediction_length + 1, prediction_length):
        #set_all_seeds(seed)
        # end -> donde termina la predicción 
        end = start + prediction_length
        # start_context -> marca el inicio del contexto  
        start_context = max(0, start - context_len)
        # context_series -> contexto con el que predecir
        context_series = full_df.iloc[start_context:start] 
        # Realizamos la predicción con el modelo ya entrenado (predictor) y almacenamos el tiempo 
        tracker = OfflineEmissionsTracker(
            project_name="Chronos_experiments",
            measure_power_secs=10,
            country_iso_code="ESP",
            tracking_mode="process",
            log_level="error",
            save_to_file=False
        )
        tracker.start()
        start_pred_time = time.time()

        sales_pred_df = pipeline.predict_df(
            context_series,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column=date,
            target="target",
        )

        tiempo = time.time() - start_pred_time
        inference_emissions = tracker.stop()
        inference_time = inference_time+tiempo 
        inference_emissions_count= inference_emissions_count+inference_emissions

        # Tomamos la media como valor puntual predicho, tomamos los valores reales y calculamos la bondad de ajuste 
        
        pred_val = sales_pred_df["predictions"].iloc[0]
        true_val = full_series[start:end]

        rango_target = max_target - min_target
        pred_val_desescalado = pred_val * rango_target + min_target

        pred_list.append(pred_val_desescalado)
        real_list.append(true_val[0])
        
    pred_col_name = f"pred_Chronos2"

    pred_vs_real_df = pd.DataFrame({
        "real": real_list,
        pred_col_name: pred_list
    })

    return pred_vs_real_df, inference_time, inference_emissions_count

# Función para crear ventanado de entrenamiento 
def crear_ventanas(series, context_len, prediction_length, col_target):
    X, y = [], []
    for i in range(len(series) - context_len - prediction_length + 1):
        x_ventana = series.iloc[i:i+context_len]  # contexto (DataFrame o Serie)
        y_ventana = series.iloc[i+context_len:i+context_len+prediction_length][col_target].values
        X.append(x_ventana)
        y.append(y_ventana)

    X = np.asarray(X)
    y = np.asarray(y)

    # Aplanar X a 2D para sklearn
    if X.ndim == 3:  # (n, t, f)
        n, t, f = X.shape
        X = X.reshape(n, t * f)
    elif X.ndim == 2:  # (n, t) si solo hay 1 feature
        n, t = X.shape
        X = X.reshape(n, t)

    # y: 1D si prediction_length == 1
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    return X, y

def evaluar_modelo_sklearn_expanding(nombre_dataset, df, frecuencia, seed,
                                     modelo_sklearn,
                                     context_len=30,
                                     prediction_length=1,
                                     initial_train_size=0.8,
                                     target="target", date="date", item_id_col=None):
    df = df.dropna(subset=[target])
    df = df.dropna(axis=1)           # eliminamos columnas que sigan teniendo NaN
    df = df.drop(columns=date)
    full_series = df[target].values
    total_len = len(full_series)
    initial_len = int(initial_train_size * total_len)

    past_covariates_cols = [
        col for col in df.columns
        if col not in {target, item_id_col, "item_id", date, "date"}
    ]


    train_df = df.iloc[:initial_len]

    # Escalamos con MinMax el conjunto de entrenamiento y aplicamos la transformación al resto de la serie 
    columns_to_scale = past_covariates_cols+[target] 
    scaler = MinMaxScaler().fit(train_df[columns_to_scale])
    train_df[columns_to_scale] = scaler.transform(train_df[columns_to_scale])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    params = pd.DataFrame({
        'min_dato': scaler.data_min_,
        'max_dato': scaler.data_max_,
        'rango': scaler.data_range_,
        'offset_usado': scaler.min_,
        'factor_escala': scaler.scale_
    }, index=columns_to_scale)


    min_target = params.loc[target, 'min_dato']
    max_target = params.loc[target, 'max_dato']

    # Crear dataset supervisado
    X_train, y_train = crear_ventanas(train_df, context_len, prediction_length, target)
    n_feats = X_train.shape[1]
    print(f"[4] Ventanas entrenamiento -> X_train:{X_train.shape}, y_train:{y_train.shape}")

    #Hyperparámetros
    if modelo_sklearn == "RandomForestRegressor":
        modelo_sklearn=RandomForestRegressor(n_estimators=100,random_state=seed)
        hyperparams = {
            "context_length": context_len,
            "estimator": "RandomForestRegressor",
            "param_distributions": {
                "max_depth": Categorical([None, 5, 10, 15]),
                "max_features": Categorical(["sqrt", "log2", 0.5, 0.8, 1.0]),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 8),
                "bootstrap": Categorical([True,False]),
            },
        }

    elif modelo_sklearn == "HistGradientBoostingRegressor":
        modelo_sklearn=HistGradientBoostingRegressor(random_state=seed)
        hyperparams = {
            "context_length": context_len,
            "estimator": "HistGradientBoostingRegressor",
            "param_distributions": {
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "max_depth": Categorical([3, 6, 9, 12, None]),
                "l2_regularization": Real(0.0, 0.2, prior="uniform"),
                "max_leaf_nodes": Integer(15, 255),
                "min_samples_leaf": Integer(5, 100),
            },
        }

    elif modelo_sklearn == "LinearRegression":
        modelo_sklearn=LinearRegression()
        hyperparams = {
            "context_length": context_len,
            "estimator": "LinearRegression",
            "param_distributions": {
                "fit_intercept": Categorical([True, False]),
            },
        }

    elif modelo_sklearn == "Lasso":
        modelo_sklearn=Lasso(random_state=seed)
        hyperparams = {
            "context_length": context_len,
            "estimator": "Lasso",
            "param_distributions": {
                "alpha": Real(1e-4, 10, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
                "positive": Categorical([False, True]),
            },
        }

    elif modelo_sklearn == "Ridge":
        modelo_sklearn=Ridge(random_state=seed)
        hyperparams = {
            "context_length": context_len,
            "estimator": "Ridge",
            "param_distributions": {
                "alpha": Real(1e-4, 1e3, prior="log-uniform"),
                "fit_intercept": Categorical([True, False]),
            },
        }

    elif modelo_sklearn == "KNeighborsRegressor":
        modelo_sklearn=KNeighborsRegressor()
        hyperparams = {
            "context_length": context_len,
            "estimator": "KNeighborsRegressor",
            "param_distributions": {
                "n_neighbors": Integer(5, 15),
                "p": Categorical([1, 2]),
            },
        }

    else:
        hyperparams = {modelo_sklearn: {}}
        
    # --- Entrenamiento ---
    print(f"\n📊 === Entrenando modelo con {len(X_train)} ventanas ===")
    modelo = clone(modelo_sklearn)
    if "param_distributions" in hyperparams and hyperparams["param_distributions"] is not None:
        modelo = BayesSearchCV(
            modelo,
            hyperparams["param_distributions"],
            n_iter=20,
            cv=3,
            n_jobs=-1,
            random_state=seed
        )

        tracker = OfflineEmissionsTracker(
            project_name="Chronos_experiments",
            measure_power_secs=10,
            country_iso_code="ESP",
            tracking_mode="process",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        start_train_time = time.time()

        modelo.fit(X_train, y_train)
        training_time = time.time() - start_train_time # Forma algo "artesanal" de tomar el tiempo de entrenamiento 
        train_emissions = tracker.stop()

        print(f"[5] Modelo entrenado. n_features_in_={modelo.best_estimator_.n_features_in_}")
    else:
        tracker = OfflineEmissionsTracker(
            project_name="Chronos_experiments",
            measure_power_secs=10,
            country_iso_code="ESP",
            tracking_mode="process",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        start_train_time = time.time()

        modelo.fit(X_train, y_train)
        training_time = time.time() - start_train_time # Forma algo "artesanal" de tomar el tiempo de entrenamiento 
        train_emissions = tracker.stop()

        print(f"[5] Modelo entrenado. n_features_in_={modelo.n_features_in_}")

    # --- Backtesting ---

    pred_list = []
    real_list = []
    inference_time = 0
    inference_emissions_count= 0 

    print(f"\n📊 === Backtesting: {nombre_dataset} ===")
    for start in range(initial_len, total_len - prediction_length + 1, prediction_length):
        #set_all_seeds(seed)
         # end -> donde termina la predicción 
        end = start + prediction_length
        # start_context -> marca el inicio del contexto  
        start_context = max(0, start - context_len)
        # context_series -> contexto con el que predecir
        context_series = df.iloc[start_context:start] 
        X_pred = context_series.to_numpy().reshape(1, -1)   # 1 muestra, context_len * n_features
        
        print(f"    - context_series shape: {context_series.shape}")
        print(f"    - X_pred shape: {X_pred.shape}")
        print(f"    - modelo espera: {n_feats} features")

        tracker = OfflineEmissionsTracker(
            project_name="Chronos_experiments",
            measure_power_secs=10,
            country_iso_code="ESP",
            tracking_mode="process",
            log_level="error",
            save_to_file=False
        )
        tracker.start()
        start_pred_time = time.time()
        pred_val = modelo.predict(X_pred).flatten()
        tiempo = time.time() - start_pred_time
        inference_emissions = tracker.stop()
        inference_time = inference_time+tiempo 
        inference_emissions_count= inference_emissions_count+inference_emissions



        true_val = full_series[start:end]
        val_anterior = full_series[start-1:end-1]
        rango_target = max_target - min_target
        pred_val_desescalado = pred_val[0] * rango_target + min_target

        pred_list.append(pred_val_desescalado)
        real_list.append(true_val[0])

    # --- Resultados ---
    nombre_modelo = type(modelo_sklearn).__name__

    pred_col_name = f"pred_{nombre_modelo}"

    pred_vs_real_df = pd.DataFrame({
        "real": real_list,
        pred_col_name: pred_list
    })

    return pred_vs_real_df, training_time, train_emissions, inference_time, inference_emissions_count

def test_accion(nombre_accion, dataframe=df): 
    seed_pair = seed_from_pair(nombre_accion,"TemporalFusionTransformer", SEED)
    set_all_seeds(seed_pair)
    df1, training_time1, train_emissions1, inference_time1, inference_emissions_count1 = evaluar_modelo_autogluon(nombre_accion, dataframe, "D", "TemporalFusionTransformer", seed=seed_pair,
                          initial_train_size=0.8, prediction_length=1, context_len=30, evalM="SMAPE",
                          item_id_col="id", target=nombre_accion, date="Date")
    seed_pair = seed_from_pair(nombre_accion,"PatchTST", SEED)
    set_all_seeds(seed_pair)
    df2, training_time2, train_emissions2, inference_time2, inference_emissions_count2 = evaluar_modelo_autogluon(nombre_accion, dataframe, "D", "PatchTST", seed=seed_pair,
                          initial_train_size=0.8, prediction_length=1, context_len=30, evalM="SMAPE",
                          item_id_col="id", target=nombre_accion, date="Date")
    seed_pair = seed_from_pair(nombre_accion,"DeepAR", SEED)
    set_all_seeds(seed_pair)
    df3, training_time3, train_emissions3, inference_time3, inference_emissions_count3 = evaluar_modelo_autogluon(nombre_accion, dataframe, "D", "DeepAR", seed=seed_pair,
                          initial_train_size=0.8, prediction_length=1, context_len=30, evalM="SMAPE",
                          item_id_col="id", target=nombre_accion, date="Date")
    seed_pair = seed_from_pair(nombre_accion,"Chronos2", SEED)
    set_all_seeds(seed_pair)
    df4, inference_time4, inference_emissions_count4 = evaluar_modelo_chronos2(nombre_accion, dataframe, "D", seed=seed_pair,
                          initial_train_size=0.8, prediction_length=1, context_len=30, evalM="SMAPE",
                          item_id_col="id", target=nombre_accion, date="Date")
    seed_pair = seed_from_pair(nombre_accion,"KNeighborsRegressor", SEED)
    set_all_seeds(seed_pair)
    df5, training_time5, train_emissions5, inference_time5, inference_emissions_count5 = evaluar_modelo_sklearn_expanding(nombre_accion, dataframe, "D", seed_pair,
                                     "KNeighborsRegressor",
                                     context_len=30,
                                     prediction_length=1,
                                     initial_train_size=0.8,
                                     target=nombre_accion, date="Date", item_id_col=None)
    seed_pair = seed_from_pair(nombre_accion,"HistGradientBoostingRegressor", SEED)
    set_all_seeds(seed_pair)
    df6, training_time6, train_emissions6, inference_time6, inference_emissions_count6 = evaluar_modelo_sklearn_expanding(nombre_accion, dataframe, "D", seed_pair,
                                     "HistGradientBoostingRegressor",
                                     context_len=30,
                                     prediction_length=1,
                                     initial_train_size=0.8,
                                     target=nombre_accion, date="Date", item_id_col=None)
    seed_pair = seed_from_pair(nombre_accion,"RandomForestRegressor", SEED)
    set_all_seeds(seed_pair)
    df7, training_time7, train_emissions7, inference_time7, inference_emissions_count7 = evaluar_modelo_sklearn_expanding(nombre_accion, dataframe, "D", seed_pair,
                                     "RandomForestRegressor",
                                     context_len=30,
                                     prediction_length=1,
                                     initial_train_size=0.8,
                                     target=nombre_accion, date="Date", item_id_col=None)
    seed_pair = seed_from_pair(nombre_accion,"LinearRegression", SEED)
    set_all_seeds(seed_pair)
    df8, training_time8, train_emissions8, inference_time8, inference_emissions_count8 = evaluar_modelo_sklearn_expanding(nombre_accion, dataframe, "D", seed_pair,
                                     "LinearRegression",
                                     context_len=30,
                                     prediction_length=1,
                                     initial_train_size=0.8,
                                     target=nombre_accion, date="Date", item_id_col=None)
    seed_pair = seed_from_pair(nombre_accion,"Ridge", SEED)
    set_all_seeds(seed_pair)
    df9, training_time9, train_emissions9, inference_time9, inference_emissions_count9 = evaluar_modelo_sklearn_expanding(nombre_accion, dataframe, "D", seed_pair,
                                     "Ridge",
                                     context_len=30,
                                     prediction_length=1,
                                     initial_train_size=0.8,
                                     target=nombre_accion, date="Date", item_id_col=None) 
    seed_pair = seed_from_pair(nombre_accion,"Lasso", SEED)
    set_all_seeds(seed_pair)  
    df10, training_time10, train_emissions10, inference_time10, inference_emissions_count10 = evaluar_modelo_sklearn_expanding(nombre_accion, dataframe, "D", seed_pair,
                                     "Lasso",
                                     context_len=30,
                                     prediction_length=1,
                                     initial_train_size=0.8,
                                     target=nombre_accion, date="Date", item_id_col=None)   

    df_total = pd.concat([df1.set_index("real"), 
                      df2.set_index("real"), 
                      df3.set_index("real"), 
                      df4.set_index("real"), 
                      df5.set_index("real"), 
                      df6.set_index("real"), 
                      df7.set_index("real"), 
                      df8.set_index("real"), 
                      df9.set_index("real"), 
                      df10.set_index("real")], axis=1) 
    meta_rows = [
        {
            "accion": nombre_accion,
            "modelo": "TemporalFusionTransformer",
            "training_time": training_time1,
            "train_emissions": train_emissions1,
            "inference_time": inference_time1,
            "inference_emissions": inference_emissions_count1,
        },
        {
            "accion": nombre_accion,
            "modelo": "PatchTST",
            "training_time": training_time2,
            "train_emissions": train_emissions2,
            "inference_time": inference_time2,
            "inference_emissions": inference_emissions_count2,
        },
        {
            "accion": nombre_accion,
            "modelo": "DeepAR",
            "training_time": training_time3,
            "train_emissions": train_emissions3,
            "inference_time": inference_time3,
            "inference_emissions": inference_emissions_count3,
        },
        {
            "accion": nombre_accion,
            "modelo": "Chronos2",
            "training_time": 0,
            "train_emissions": 0,
            "inference_time": inference_time4,
            "inference_emissions": inference_emissions_count4,
        },
        {
            "accion": nombre_accion,
            "modelo": "KNeighborsRegressor",
            "training_time": training_time5,
            "train_emissions": train_emissions5,
            "inference_time": inference_time5,
            "inference_emissions": inference_emissions_count5,
        },
        {
            "accion": nombre_accion,
            "modelo": "HistGradientBoostingRegressor",
            "training_time": training_time6,
            "train_emissions": train_emissions6,
            "inference_time": inference_time6,
            "inference_emissions": inference_emissions_count6,
        },
        {
            "accion": nombre_accion,
            "modelo": "RandomForestRegressor",
            "training_time": training_time7,
            "train_emissions": train_emissions7,
            "inference_time": inference_time7,
            "inference_emissions": inference_emissions_count7,
        },
        {
            "accion": nombre_accion,
            "modelo": "LinearRegression",
            "training_time": training_time8,
            "train_emissions": train_emissions8,
            "inference_time": inference_time8,
            "inference_emissions": inference_emissions_count8,
        },
        {
            "accion": nombre_accion,
            "modelo": "Ridge",
            "training_time": training_time9,
            "train_emissions": train_emissions9,
            "inference_time": inference_time9,
            "inference_emissions": inference_emissions_count9,
        },
        {
            "accion": nombre_accion,
            "modelo": "Lasso",
            "training_time": training_time10,
            "train_emissions": train_emissions10,
            "inference_time": inference_time10,
            "inference_emissions": inference_emissions_count10,
        }
    ]

    df_meta = pd.DataFrame(meta_rows)

    return df_total, df_meta

def rangos_por_accion_en_5(df, accion, date_col="Date"):
    """
    Calcula 5 rangos de fechas basados EXCLUSIVAMENTE en la serie 'accion':
    - Ordena por fecha.
    - Elimina NaN SOLO en la columna 'accion'.
    - Divide esa serie en 5 trozos ~iguales y devuelve [(ini, fin), ...] por cada trozo.
    """
    s = df[[date_col, accion]].dropna(subset=[accion]).copy()
    if s.empty:
        return [(None, None)] * 5
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.sort_values(date_col).reset_index(drop=True)

    # Cortes por índice de la serie válida de la acción
    idx_parts = np.array_split(np.arange(len(s)), 5)
    rangos = []
    for idx in idx_parts:
        if len(idx) == 0:
            rangos.append((None, None))
        else:
            sub = s.iloc[idx]
            rangos.append((sub[date_col].min(), sub[date_col].max()))
    return rangos

def test_accion_particionado_por_accion(nombre_accion, df, date_col="Date", out_dir="ResultadosBaseDatos"):
    """
    1) Calcula 5 rangos temporales usando SOLO la serie de 'nombre_accion' (sin NaN).
    2) Para cada rango, filtra el DataFrame completo por fecha (para conservar todas las columnas),
       y vuelve a quitar NaN del target por seguridad.
    3) Evalúa los modelos y guarda 5 CSV (uno por partición) en: {out_dir}/{nombre_accion}/...
    Devuelve lista de DataFrames de resultados.
    """
    os.makedirs(os.path.join(out_dir, nombre_accion), exist_ok=True)

    # Asegurar orden temporal en el df base
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    rangos = rangos_por_accion_en_5(df, nombre_accion, date_col=date_col)
    resultados = []

    for i, (ini, fin) in enumerate(rangos, start=1):
        if ini is None or fin is None:
            print(f"⚠️ Partición {i} de {nombre_accion} está vacía (la serie no tenía datos suficientes).")
            resultados.append(pd.DataFrame())
            continue

        # Filtrar el DF completo por el rango (particionamiento guiado por la acción)
        df_part = df[(df[date_col] >= ini) & (df[date_col] <= fin)].copy()

        # Evaluación
        df_res, df_meta = test_accion(nombre_accion, df_part)
        df_res["particion"] = i
        df_res["rango_particion"] = f"{ini.strftime('%Y%m%d')}-{fin.strftime('%Y%m%d')}"
        df_meta["particion"] = i
        df_meta["rango_particion"] = f"{ini.strftime('%Y%m%d')}-{fin.strftime('%Y%m%d')}"

        # Guardar CSV
        carpeta = os.path.join(out_dir, nombre_accion)
        nombre = f"{nombre_accion}_part{i}_{ini.strftime('%Y%m%d')}-{fin.strftime('%Y%m%d')}.csv"
        ruta_csv = os.path.join(carpeta, nombre)
        df_res.to_csv(ruta_csv, index=True)
        print(f"✅ [{nombre_accion}] Partición {i} guardada en: {ruta_csv}")

        nombre2 = f"{nombre_accion}_metadata_part{i}_{ini.strftime('%Y%m%d')}-{fin.strftime('%Y%m%d')}.csv"
        ruta_csv2 = os.path.join(carpeta, nombre2)
        df_meta.to_csv(ruta_csv2, index=True)
        print(f"✅ [{nombre_accion}] Datos meta partición {i} guardada en: {ruta_csv2}")

columnasIbex = ["ACS"]

if __name__ == "__main__": 
    for accion in columnasIbex:
        test_accion_particionado_por_accion(accion, df, date_col="Date")