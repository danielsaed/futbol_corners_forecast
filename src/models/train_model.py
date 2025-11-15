import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor
import joblib


class TRAIN_MODEL():
    def __init__(self, nombre, use_grid_search=False, config_path="config/model_config.json"):
        """
        Entrenar modelo con tracking MLflow
        
        Args:
            nombre: Identificador del modelo (ej: "v3_production")
            use_grid_search: True = buscar hiperparámetros, False = usar config guardado
            config_path: Ruta al archivo de configuración con hiperparámetros
        """
        # ===========================
        # CONFIGURACIÓN MLFLOW
        # ===========================
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("corners_prediction")
        
        self.nombre = nombre
        self.use_grid_search = use_grid_search
        self.config_path = config_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Iniciar run de MLflow
        with mlflow.start_run(run_name=f"{nombre}_{self.timestamp}") as run:
            self.run_id = run.info.run_id
            
            print(f"\n{'='*80}")
            print(f"🚀 Entrenamiento iniciado con MLflow")
            print(f"   Run ID: {self.run_id}")
            print(f"   Nombre: {nombre}")
            print(f"   GridSearch: {'SÍ' if use_grid_search else 'NO (usando config)'}")
            print(f"{'='*80}\n")
            
            # Tags básicos
            mlflow.set_tags({
                "model_name": nombre,
                "timestamp": self.timestamp,
                "grid_search_used": str(use_grid_search),
                "framework": "XGBoost",
                "task": "regression"
            })
            
            # Pipeline de entrenamiento
            try:
                self.init_variables()
                self.load_dataset()
                self.split_train_test(0.15)
                self.define_model()
                
                if use_grid_search:
                    print("🔍 Ejecutando GridSearch (puede tardar)...")
                    self.train_grid_search()
                    self.save_best_params()  # Guardar para futuros entrenamientos
                else:
                    print("⚡ Usando hiperparámetros guardados (rápido)")
                    self.load_best_params()
                
                self.train_model()
                self.test_and_eval()
                self.top_features()
                self.save_models(nombre)
                
                mlflow.set_tag("status", "SUCCESS")
                print(f"\n✅ Entrenamiento completado")
                print(f"📊 Ver en MLflow UI: mlflow ui")
                
            except Exception as e:
                mlflow.set_tag("status", "FAILED")
                print(f"\n❌ Error: {e}")
                raise

    def init_variables(self):
        """Definir espacio de búsqueda para GridSearch"""
        # ✅ GRID INTELIGENTE (~243 combinaciones = 1-3 horas)
        self.param_grid = {
            'n_estimators': [200],              # 1 valor (200 suele ser óptimo)
            'max_depth': [3, 4, 5],             # 3 valores (clave)
            'learning_rate': [0.02, 0.03],      # 2 valores (0.01 es muy lento)
            'reg_alpha': [3.0, 5.0],            # 2 valores
            'reg_lambda': [5.0, 8.0],           # 2 valores
            'gamma': [0.5, 1.0],                # 2 valores
            'subsample': [0.7],                 # 1 valor (0.7 suele funcionar)
            'colsample_bytree': [0.7],          # 1 valor
            'colsample_bylevel': [0.6],         # 1 valor
            'min_child_weight': [5, 7]          # 2 valores
        }
        # Combinaciones: 1 × 3 × 2 × 2 × 2 × 2 × 1 × 1 × 1 × 2 = 192
        # Tiempo: ~1.5-3 horas ⏱️
        
        # Loggear configuración del grid
        if self.use_grid_search:
            for param, values in self.param_grid.items():
                mlflow.log_param(f"grid_{param}", str(values))
        
        print("✅ Variables inicializadas")

    def load_dataset(self):
        """Cargar y preparar dataset"""
        
        self.df_data = pd.read_csv("dataset/processed/dataset_processed.csv")
        self.y = self.df_data["y"]
        self.df_data = self.df_data.drop(["y"], axis=1)
        self.y_array = np.array(self.y).flatten()
        
        # Filtrar outliers (3-17 corners)
        mask = (self.y_array >= 3) & (self.y_array <= 17)
        self.df_data = self.df_data[mask].copy()
        self.y_array = self.y_array[mask]
        
        # Limpiar nulos
        if self.df_data.isnull().any().any():
            self.df_data = self.df_data.fillna(0)
        
        # Loggear info del dataset
        mlflow.log_params({
            "dataset_samples": len(self.df_data),
            "dataset_features": self.df_data.shape[1],
            "target_min": float(self.y_array.min()),
            "target_max": float(self.y_array.max()),
            "target_mean": float(self.y_array.mean()),
            "target_std": float(self.y_array.std())
        })
        
        print(f"✅ Dataset cargado: {self.df_data.shape}")

    def split_train_test(self, test_size_):
        """Dividir datos en train/val/test"""
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df_data, self.y_array, 
            test_size=test_size_, 
            random_state=42, 
            shuffle=True
        )
        
        # Escalar
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), 
            columns=self.X_train.columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test), 
            columns=self.X_test.columns
        )
        
        # Split validación
        self.X_train_fit, self.X_val, self.y_train_fit, self.y_val = train_test_split(
            self.X_train, self.y_train, 
            test_size=0.15, 
            random_state=43
        )
        
        # Loggear splits
        mlflow.log_params({
            "train_samples": len(self.X_train_fit),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "test_size": test_size_
        })
        
        print(f"✅ Train: {len(self.X_train_fit)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}")

    def define_model(self):
        """Definir modelo base y GridSearch"""
        
        self.xgb_base = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        if self.use_grid_search:
            self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            self.mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            
            self.grid_search = GridSearchCV(
                estimator=self.xgb_base,
                param_grid=self.param_grid,
                cv=self.kfold,
                scoring=self.mae_scorer,
                n_jobs=-1,
                verbose=2,
                return_train_score=True
            )

    def train_grid_search(self):
        """Ejecutar GridSearch y guardar mejores params"""
        
        print("\n🔍 Buscando mejores hiperparámetros...")
        self.grid_search.fit(self.X_train_fit, self.y_train_fit)
        
        # Mejores parámetros
        self.best_params = self.grid_search.best_params_
        
        # Loggear en MLflow
        for param, value in self.best_params.items():
            mlflow.log_param(f"best_{param}", value)
        
        mlflow.log_metric("cv_best_mae", -self.grid_search.best_score_)
        
        print(f"\n✅ Mejores hiperparámetros encontrados:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        print(f"   CV MAE: {-self.grid_search.best_score_:.4f}")

    def save_best_params(self):
        """Guardar mejores hiperparámetros en archivo JSON"""
        
        os.makedirs("config", exist_ok=True)
        
        config = {
            "model_name": self.nombre,
            "timestamp": self.timestamp,
            "best_params": self.best_params,
            "cv_mae": float(-self.grid_search.best_score_),
            "run_id": self.run_id
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Loggear archivo en MLflow
        mlflow.log_artifact(self.config_path)
        
        print(f"💾 Hiperparámetros guardados en: {self.config_path}")

    def load_best_params(self):
        """Cargar hiperparámetros desde archivo JSON"""
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"No se encontró {self.config_path}. "
                "Ejecuta primero con use_grid_search=True"
            )
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.best_params = config["best_params"]
        
        # Loggear params en MLflow
        for param, value in self.best_params.items():
            mlflow.log_param(f"loaded_{param}", value)
        
        mlflow.log_param("config_source", self.config_path)
        mlflow.log_param("previous_cv_mae", config.get("cv_mae", "N/A"))
        
        print(f"✅ Hiperparámetros cargados desde: {self.config_path}")
        print(f"   Origen: {config.get('model_name', 'unknown')} ({config.get('timestamp', 'unknown')})")

    def train_model(self):
        """Entrenar modelo final con mejores params"""
        
        self.xgb_model = XGBRegressor(
            **self.best_params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        self.xgb_model.fit(
            self.X_train_fit, 
            self.y_train_fit,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        
        print("✅ Modelo entrenado")

    def test_and_eval(self):
        """Evaluar y loggear métricas"""
        
        # Predicciones
        y_train_pred = self.xgb_model.predict(self.X_train_fit)
        y_val_pred = self.xgb_model.predict(self.X_val)
        y_test_pred = self.xgb_model.predict(self.X_test)
        
        # Calcular métricas
        metrics = {
            'train': {
                'mae': mean_absolute_error(self.y_train_fit, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_train_fit, y_train_pred)),
                'r2': r2_score(self.y_train_fit, y_train_pred)
            },
            'val': {
                'mae': mean_absolute_error(self.y_val, y_val_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
                'r2': r2_score(self.y_val, y_val_pred)
            },
            'test': {
                'mae': mean_absolute_error(self.y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'r2': r2_score(self.y_test, y_test_pred)
            }
        }
        
        # Loggear TODAS las métricas en MLflow
        for set_name, set_metrics in metrics.items():
            for metric_name, value in set_metrics.items():
                mlflow.log_metric(f"{set_name}_{metric_name}", value)
        
        # Cross-validation
        cv_mae = cross_val_score(
            self.xgb_model, self.X_train, self.y_train, 
            cv=5, scoring='neg_mean_absolute_error'
        )
        cv_r2 = cross_val_score(
            self.xgb_model, self.X_train, self.y_train, 
            cv=5, scoring='r2'
        )
        
        mlflow.log_metric("cv_mae_mean", -cv_mae.mean())
        mlflow.log_metric("cv_mae_std", cv_mae.std())
        mlflow.log_metric("cv_r2_mean", cv_r2.mean())
        mlflow.log_metric("cv_r2_std", cv_r2.std())
        
        # Análisis de errores
        test_errors = np.abs(self.y_test - y_test_pred)
        mlflow.log_metric("test_error_median", float(np.median(test_errors)))
        mlflow.log_metric("test_error_p90", float(np.percentile(test_errors, 90)))
        mlflow.log_metric("test_pct_error_lt_2", float((test_errors < 2.0).sum() / len(test_errors) * 100))
        
        # Gap de overfitting
        gap = metrics['train']['r2'] - metrics['test']['r2']
        mlflow.log_metric("overfitting_gap", gap)
        
        print(f"\n📊 MÉTRICAS:")
        print(f"   Train MAE: {metrics['train']['mae']:.4f} | R²: {metrics['train']['r2']:.4f}")
        print(f"   Val   MAE: {metrics['val']['mae']:.4f} | R²: {metrics['val']['r2']:.4f}")
        print(f"   Test  MAE: {metrics['test']['mae']:.4f} | R²: {metrics['test']['r2']:.4f}")
        print(f"   CV    MAE: {-cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
        print(f"   Overfitting Gap: {gap:.4f}")

    def top_features(self):
        """Guardar importancia de features"""
        
        feature_importance = pd.DataFrame({
            'feature': self.df_data.columns,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Guardar CSV
        feature_importance.to_csv(f"models/feature_importance_{self.nombre}.csv", index=False)
        mlflow.log_artifact(f"models/feature_importance_{self.nombre}.csv")
        
        # Loggear top 10
        for idx, row in feature_importance.head(10).iterrows():
            mlflow.log_metric(f"feat_imp_{row['feature']}", row['importance'])
        
        print(f"\n🔍 Top 5 features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

    def save_models(self, nombre):
        """Guardar modelos localmente y en MLflow"""
        
        os.makedirs("models", exist_ok=True)
        
        # Paths
        model_path = f'models/xgboost_corners_{nombre}.pkl'
        scaler_path = f'models/scaler_corners_{nombre}.pkl'
        
        # Guardar archivos
        joblib.dump(self.xgb_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Loggear en MLflow
        mlflow.xgboost.log_model(
            self.xgb_model,
            artifact_path="model",
            registered_model_name=f"corners_predictor"
        )
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        
        print(f"\n💾 Modelos guardados:")
        print(f"   {model_path}")
        print(f"   {scaler_path}")
        print(f"   MLflow Model Registry ✓")


# ===========================
# USO
# ===========================

if __name__ == "__main__":
    
    # ========================================
    # OPCIÓN 1: Primera vez o cada 3-6 meses
    # Ejecutar GridSearch (LENTO, 30-60 min)
    # ========================================
    # model = TRAIN_MODEL(
    #     nombre="v4_grid_search",
    #     use_grid_search=True  # Busca mejores hiperparámetros
    # )
    
    # ========================================
    # OPCIÓN 2: Reentrenamiento regular
    # Usar hiperparámetros guardados (RÁPIDO, 2-5 min)
    # ========================================
    model = TRAIN_MODEL(
        nombre="v4_retrain",
        use_grid_search=True  # Usa config/model_config.json
    )