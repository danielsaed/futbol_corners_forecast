import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV,  # ✅ CAMBIO: Usaremos GridSearch para búsqueda exhaustiva
    cross_val_score,
    KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

from xgboost import XGBRegressor
import joblib


# Métricas por set
def calc_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'set': set_name,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }



class TRAIN_MODEL():
    def __init__(self):
        self.init_variables()
        self.load_dataset()
        self.split_train_test(.15)
        self.define_model()
        self.train_grid_search()
        self.train_model()
        self.test_and_eval()
        self.top_features()
        self.save_models()

    def init_variables(self):
        print("")

        self.param_grid = {
            # ⬇️ REDUCIR COMPLEJIDAD
            'n_estimators': [150, 200],  # Menos árboles
            'max_depth': [3, 4],  # ⬇️ Árboles más simples
            'learning_rate': [0.02, 0.03],  # ⬇️ Aprendizaje MÁS lento
            
            # ⬆️ AUMENTAR REGULARIZACIÓN
            'reg_alpha': [3.0, 5.0],  # ⬆️ L1 más fuerte
            'reg_lambda': [5.0, 8.0],  # ⬆️ L2 más fuerte
            'gamma': [0.5, 1.0],  # ⬆️ Penalización más alta
            
            # ⬇️ REDUCIR SAMPLING
            'subsample': [0.6, 0.7],  # ⬇️ Menos datos por árbol
            'colsample_bytree': [0.6, 0.7],  # ⬇️ Menos features por árbol
            'colsample_bylevel': [0.6],  # ⬇️ Menos features por nivel
            
            # ⬆️ AUMENTAR TAMAÑO MÍNIMO
            'min_child_weight': [5, 7]  # ⬆️ Hojas más grandes
        }
        print("Varible loaded")

    def load_dataset(self):
        
        self.df_data = pd.read_csv("dataset\processed\dataset_processed.csv")
        self.y = self.df_data["y"]
        self.df_data = self.df_data.drop(["y"],axis=1)

        self.y_array = np.array(self.y).flatten()

        mask = (self.y_array >= 4) & (self.y_array <= 16)

        # Aplicar filtro
        self.df_data = self.df_data[mask].copy()
        self.y_array = self.y_array[mask]

                # Verificar y limpiar datos
        assert len(self.df_data) == len(self.y_array), f"❌ ERROR: Dimensiones no coinciden"

        if self.df_data.isnull().any().any():
            print(f"⚠️ Hay {self.df_data.isnull().sum().sum()} valores nulos en X")
            self.df_data = self.df_data.fillna(0)

        if np.isnan(self.y_array).any():
            mask = ~np.isnan(self.y_array)
            self.df_data = self.df_data[mask]
            self.y_array = self.y_array[mask]

        print("Dataset Loaded")

    def split_train_test(self,test_size_):
        print("")

        self.X_scaled = self.df_data

        # ✅ CAMBIO: Aumentamos train al 80% para tener más datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, 
            self.y_array, 
            test_size=test_size_,  # ⬇️ REDUCIR de 0.30 a 0.20
            random_state=42,
            shuffle=True
        )

        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)  # Solo transform

        # ✅ NUEVO: Split adicional para validación temprana
        self.X_train_fit, self.X_val, self.y_train_fit, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=test_size_,  # 20% del train para validación
            random_state=43
        )

        print(f"\n📊 División de datos:")
        print(f"   Train (fit):      {self.X_train_fit.shape[0]} muestras ({self.X_train_fit.shape[0]/len(self.X_scaled)*100:.1f}%)")
        print(f"   Validation:       {self.X_val.shape[0]} muestras ({self.X_val.shape[0]/len(self.X_scaled)*100:.1f}%)")
        print(f"   Test (hold-out):  {self.X_test.shape[0]} muestras ({self.X_test.shape[0]/len(self.X_scaled)*100:.1f}%)")

    def define_model(self):
        
        # ✅ MODELO BASE CON CONFIGURACIÓN CONSERVADORA
        self.xgb_base = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0  # Silenciar warnings
        )

        # ✅ CROSS-VALIDATION CON K-FOLD ESTRATIFICADO
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # ⬇️ 3 folds en lugar de 5

        # ✅ SCORER PERSONALIZADO (priorizar MAE sobre R²)
        self.mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        # ✅ GRIDSEARCH CON CONFIGURACIÓN ANTI-OVERFITTING
        self.grid_search = GridSearchCV(
            estimator=self.xgb_base,
            param_grid=self.param_grid,
            cv=self.kfold,  # 3-fold
            scoring=self.mae_scorer,
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
            error_score='raise'
        )

        print("model defined")

    def train_grid_search(self):

        # ENTRENAR GRIDSEARCH
        self.grid_search.fit(self.X_train_fit, self.y_train_fit)


        print("\n✅ MEJORES HIPERPARÁMETROS ENCONTRADOS:")
        print("=" * 70)
        for param, value in self.grid_search.best_params_.items():
            print(f"   {param:20s}: {value}")

        print(f"\n📊 MEJOR SCORE (CV MAE): {-self.grid_search.best_score_:.4f}")

        # ✅ DETECTAR OVERFITTING EN CV
        results_df = pd.DataFrame(self.grid_search.cv_results_)
        results_df['mean_train_mae'] = -results_df['mean_train_score']
        results_df['mean_test_mae'] = -results_df['mean_test_score']
        results_df['overfitting_gap'] = results_df['mean_train_mae'] - results_df['mean_test_mae']

        # Top 5 mejores configuraciones
        top_5 = results_df.nsmallest(5, 'mean_test_mae')[
            ['mean_train_mae', 'mean_test_mae', 'overfitting_gap', 'params']
        ]

        print("\n🏆 TOP 5 CONFIGURACIONES:")
        print("=" * 70)
        for idx, row in top_5.iterrows():
            print(f"\nRank {idx+1}:")
            print(f"  Train MAE: {row['mean_train_mae']:.4f}")
            print(f"  CV MAE:    {row['mean_test_mae']:.4f}")
            print(f"  Gap:       {row['overfitting_gap']:.4f}")
            print(f"  Params:    {row['params']}")

    def train_model(self):
        print("\n🚀 Entrenando modelo final con mejores parámetros...")

        self.xgb_model = XGBRegressor(
            **self.grid_search.best_params_,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        # ✅ ENTRENAR CON EVAL_SET (sin early_stopping_rounds en fit)
        self.xgb_model.fit(
            self.X_train_fit, 
            self.y_train_fit,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )

        # ✅ VERIFICAR SI HAY EARLY STOPPING ACTIVO
        if hasattr(self.xgb_model, 'best_iteration') and self.xgb_model.best_iteration is not None:
            print(f"✅ Modelo entrenado con early stopping en iteración: {self.xgb_model.best_iteration}")
        else:
            print(f"✅ Modelo entrenado con {self.xgb_model.n_estimators} iteraciones completas")

    def test_and_eval(self):

        # Predicciones en todos los sets
        y_train_fit_pred = self.xgb_model.predict(self.X_train_fit)
        y_val_pred = self.xgb_model.predict(self.X_val)
        y_train_pred = self.xgb_model.predict(self.X_train)  # Train completo
        y_test_pred = self.xgb_model.predict(self.X_test)    # Test hold-out

        metrics = [
            calc_metrics(self.y_train_fit, y_train_fit_pred, 'Train (Fit)'),
            calc_metrics(self.y_val, y_val_pred, 'Validation'),
            calc_metrics(self.y_train, y_train_pred, 'Train (Full)'),
            calc_metrics(self.y_test, y_test_pred, 'Test (Hold-out)')
        ]

        df_metrics = pd.DataFrame(metrics)

        print("\n📈 RESULTADOS FINALES - TODAS LAS PARTICIONES")
        print("=" * 80)
        print(df_metrics.to_string(index=False))

        # ✅ ANÁLISIS DE OVERFITTING MULTI-NIVEL
        print("\n⚠️ ANÁLISIS DE OVERFITTING:")
        print("=" * 80)

        gap_train_val = df_metrics[df_metrics['set'] == 'Train (Fit)']['r2'].values[0] - \
                        df_metrics[df_metrics['set'] == 'Validation']['r2'].values[0]

        gap_train_test = df_metrics[df_metrics['set'] == 'Train (Full)']['r2'].values[0] - \
                        df_metrics[df_metrics['set'] == 'Test (Hold-out)']['r2'].values[0]

        gap_val_test = df_metrics[df_metrics['set'] == 'Validation']['r2'].values[0] - \
                    df_metrics[df_metrics['set'] == 'Test (Hold-out)']['r2'].values[0]

        print(f"  Gap Train→Validation R²:  {gap_train_val:+.4f} {'✅ OK' if abs(gap_train_val) < 0.10 else '⚠️ OVERFITTING'}")
        print(f"  Gap Train→Test R²:        {gap_train_test:+.4f} {'✅ OK' if abs(gap_train_test) < 0.15 else '⚠️ OVERFITTING'}")
        print(f"  Gap Validation→Test R²:   {gap_val_test:+.4f} {'✅ OK' if abs(gap_val_test) < 0.05 else '⚠️ DRIFT'}")

        # ✅ CROSS-VALIDATION FINAL
        cv_scores_r2 = cross_val_score(self.xgb_model, self.X_train, self.y_train, cv=5, scoring='r2')
        cv_scores_mae = cross_val_score(self.xgb_model, self.X_train, self.y_train, cv=5, 
                                        scoring='neg_mean_absolute_error')

        print(f"\n🔄 CROSS-VALIDATION (5-FOLD) EN TRAIN COMPLETO:")
        print(f"  R² Mean:  {cv_scores_r2.mean():.4f} (±{cv_scores_r2.std()*2:.4f})")
        print(f"  MAE Mean: {-cv_scores_mae.mean():.4f} (±{cv_scores_mae.std()*2:.4f})")

        # ===========================
        # 8. ANÁLISIS DE PREDICCIONES (TEST)
        # ===========================

        test_mse = df_metrics[df_metrics['set'] == 'Test (Hold-out)']['mse'].values[0]
        test_mae = df_metrics[df_metrics['set'] == 'Test (Hold-out)']['mae'].values[0]
        test_r2 = df_metrics[df_metrics['set'] == 'Test (Hold-out)']['r2'].values[0]

        comparison = pd.DataFrame({
            'Real': self.y_test,
            'Predicho': y_test_pred,
            'Error': self.y_test - y_test_pred,
            'Error_Abs': np.abs(self.y_test - y_test_pred),
            'Error_%': (np.abs(self.y_test - y_test_pred) / np.maximum(self.y_test, 1) * 100)
        })
        comparison = comparison.sort_values('Error_Abs', ascending=False)

        print("\n🔍 PEORES PREDICCIONES (Top 5):")
        print(comparison.head(5).to_string())

        print("\n✅ MEJORES PREDICCIONES (Top 5):")
        print(comparison.tail(5).to_string())

        # Estadísticas de error
        print(f"\n📊 ESTADÍSTICAS DE ERROR (TEST):")
        print(f"  Error promedio: {comparison['Error'].mean():.2f}")
        print(f"  Error mediano: {comparison['Error'].median():.2f}")
        print(f"  Error estándar: {comparison['Error'].std():.2f}")
        print(f"  % predicciones con error <1.5: {(comparison['Error_Abs'] < 1.5).sum() / len(comparison) * 100:.1f}%")
        print(f"  % predicciones con error <2.0: {(comparison['Error_Abs'] < 2.0).sum() / len(comparison) * 100:.1f}%")
        print(f"  % predicciones con error <3.0: {(comparison['Error_Abs'] < 3.0).sum() / len(comparison) * 100:.1f}%")

        print("\n")
        print("\n" + "=" * 80)
        print("📋 RESUMEN EJECUTIVO")
        print("=" * 80)

        print(f"\n🎯 RENDIMIENTO DEL MODELO:")
        print(f"  Test R²:       {test_r2:.4f} ({'✅ EXCELENTE' if test_r2 > 0.40 else '⚠️ MEJORABLE' if test_r2 > 0.25 else '❌ BAJO'})")
        print(f"  Test MAE:      {test_mae:.4f} corners")
        print(f"  Test RMSE:     {np.sqrt(test_mse):.4f} corners")

        print(f"\n🔄 GENERALIZACIÓN:")
        print(f"  CV R² (mean):  {cv_scores_r2.mean():.4f}")
        print(f"  CV MAE (mean): {-cv_scores_mae.mean():.4f}")
        print(f"  Estabilidad:   {'✅ ALTA' if cv_scores_r2.std() < 0.05 else '⚠️ MEDIA' if cv_scores_r2.std() < 0.10 else '❌ BAJA'}")

        print(f"\n⚠️ OVERFITTING:")
        print(f"  Train-Test Gap R²: {gap_train_test:+.4f} ({'✅ ACEPTABLE' if abs(gap_train_test) < 0.15 else '⚠️ PRESENTE'})")

        print("\n" + "=" * 80)

    def top_features(self):

        print("\n🔍 TOP 20 FEATURES MÁS IMPORTANTES:")
        print("=" * 75)
        top_features = pd.DataFrame({
            'Feature': self.df_data.columns,
            'Importance': self.xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(20)

        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            bar = '█' * int(row['Importance'] * 200)
            print(f"{idx:2d}. {row['Feature']:50s} {row['Importance']:.6f} {bar}")

    def save_models(self):

        joblib.dump(self.xgb_model, r'models/xgboost_corners_optimized_test.pkl')
        joblib.dump(self.scaler, r'models/scaler_corners_xgb_test.pkl')
        joblib.dump(self.grid_search.best_params_, r'models/best_params_xgb_test.pkl')

        print("\n💾 ARCHIVOS GUARDADOS:")
        print("   ✅ xgboost_corners_optimized_v2.pkl")
        print("   ✅ scaler_corners_xgb_v2.pkl")
        print("   ✅ best_params_xgb.pkl")

        print("\n" + "=" * 80)



a = TRAIN_MODEL()