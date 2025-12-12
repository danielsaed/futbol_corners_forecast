# ===========================
# SISTEMA DE PREDICCIÓN DE CORNERS - OPTIMIZADO PARA APUESTAS (VERSIÓN COMPLETA)
# ===========================

import requests
import tempfile
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson
from scipy import stats
import os
import sys
from src.process_data.process_dataset import get_dataframes,get_head_2_head,get_points_from_result,get_team_ppp,get_ppp_difference,get_average
#from process_data.process_dataset import get_dataframes,get_head_2_head,get_points_from_result,get_team_ppp,get_ppp_difference,get_average
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
#sys.path.insert(0, project_root)
# ===========================
# 1. FUNCIONES FIABILIDAD
# ===========================

def analizar_fiabilidad_equipos(df_database, temporada="2526", min_partidos=5):
    """
    Análisis completo de fiabilidad para apuestas de corners
    No solo varianza, sino consistencia, tendencias y patrones
    """
    
    df_temp = df_database[df_database['season'] == temporada].copy()
    resultados = []
    equipos = pd.concat([df_temp['team'], df_temp['opponent']]).unique()
    
    for equipo in equipos:
        # Partidos del equipo
        partidos_equipo = df_temp[df_temp['team'] == equipo]
        
        if len(partidos_equipo) < min_partidos:
            continue
        
        ck_sacados = partidos_equipo['Pass Types_CK'].values
        
        # ===========================
        # 1. MÉTRICAS DE VARIABILIDAD
        # ===========================
        media = ck_sacados.mean()
        std = ck_sacados.std()
        cv = (std / media * 100) if media > 0 else 0
        
        # ===========================
        # 2. MÉTRICAS DE CONSISTENCIA
        # ===========================
        
        # 2.1 Porcentaje de partidos cerca de la media (±2 corners)
        cerca_media = np.sum(np.abs(ck_sacados - media) <= 2) / len(ck_sacados) * 100
        
        # 2.2 Rachas (detectar equipos con "explosiones" de corners)
        cambios_bruscos = np.sum(np.abs(np.diff(ck_sacados)) > 4)
        pct_cambios_bruscos = cambios_bruscos / (len(ck_sacados) - 1) * 100
        
        # 2.3 Cuartiles (Q1, Q2=mediana, Q3)
        q1, q2, q3 = np.percentile(ck_sacados, [25, 50, 75])
        iqr = q3 - q1  # Rango intercuartílico (más robusto que std)
        
        # ===========================
        # 3. MÉTRICAS DE TENDENCIA
        # ===========================
        
        # 3.1 Tendencia lineal (¿mejora/empeora con el tiempo?)
        jornadas = np.arange(len(ck_sacados))
        slope, intercept, r_value, p_value, std_err = stats.linregress(jornadas, ck_sacados)
        
        # 3.2 Autocorrelación (¿resultado actual predice el siguiente?)
        if len(ck_sacados) > 2:
            autocorr = np.corrcoef(ck_sacados[:-1], ck_sacados[1:])[0, 1]
        else:
            autocorr = 0
        
        # ===========================
        # 4. MÉTRICAS DE OUTLIERS
        # ===========================
        
        # 4.1 Detección de valores atípicos (método IQR)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((ck_sacados < lower_bound) | (ck_sacados > upper_bound))
        pct_outliers = outliers / len(ck_sacados) * 100
        
        # 4.2 Z-score máximo
        z_scores = np.abs(stats.zscore(ck_sacados))
        max_z = z_scores.max()
        
        # ===========================
        # 5. MÉTRICAS DE RANGO
        # ===========================
        
        rango = ck_sacados.max() - ck_sacados.min()
        rango_normalizado = rango / media if media > 0 else 0
        
        # ===========================
        # 6. SCORE GLOBAL DE FIABILIDAD
        # ===========================
        
        # Penalizaciones (0-100, menor = peor)
        score_cv = max(0, 100 - cv * 2)  # CV alto = mala
        score_consistencia = cerca_media  # Más cerca de media = mejor
        score_cambios = max(0, 100 - pct_cambios_bruscos * 2)  # Cambios bruscos = malo
        score_outliers = max(0, 100 - pct_outliers * 3)  # Outliers = malo
        score_iqr = max(0, 100 - iqr * 10)  # IQR grande = malo
        
        # Score final (promedio ponderado)
        score_fiabilidad = (
            score_cv * 0.25 +
            score_consistencia * 0.30 +
            score_cambios * 0.20 +
            score_outliers * 0.15 +
            score_iqr * 0.10
        )
        
        # ===========================
        # 7. CLASIFICACIÓN MULTI-CRITERIO
        # ===========================
        
        # Clasificación basada en score
        if score_fiabilidad >= 70:
            nivel = "EXCELENTE ⭐⭐⭐"
            color = "#27ae60"
        elif score_fiabilidad >= 55:
            nivel = "BUENO ✅"
            color = "#2ecc71"
        elif score_fiabilidad >= 40:
            nivel = "ACEPTABLE 🟡"
            color = "#f39c12"
        elif score_fiabilidad >= 25:
            nivel = "REGULAR ⚠️"
            color = "#e67e22"
        else:
            nivel = "EVITAR ⛔"
            color = "#e74c3c"
        
        resultados.append({
            'Equipo': equipo,
            'Partidos': len(ck_sacados),
            
            # Estadísticas básicas
            'Media_CK': round(media, 2),
            'Mediana_CK': round(q2, 2),
            'Std_CK': round(std, 2),
            'CV_%': round(cv, 1),
            
            # Consistencia
            'Pct_Cerca_Media': round(cerca_media, 1),
            'Cambios_Bruscos_%': round(pct_cambios_bruscos, 1),
            'IQR': round(iqr, 2),
            
            # Rango
            'Rango': int(rango),
            'Rango_Norm': round(rango_normalizado, 2),
            'Min': int(ck_sacados.min()),
            'Max': int(ck_sacados.max()),
            
            # Outliers
            'Outliers': int(outliers),
            'Pct_Outliers': round(pct_outliers, 1),
            'Max_ZScore': round(max_z, 2),
            
            # Tendencia
            'Tendencia_Slope': round(slope, 3),
            'Autocorr': round(autocorr, 3),
            
            # Score y clasificación
            'Score_Fiabilidad': round(score_fiabilidad, 1),
            'Nivel': nivel,
            'Color': color
        })
    
    df_resultado = pd.DataFrame(resultados)

    df_resultado = df_resultado.sort_values('Score_Fiabilidad', ascending=False)
    
    return df_resultado

def mostrar_analisis_fiabilidad(df_analisis, top_n=10):
    """
    Muestra el análisis completo de fiabilidad
    """
    
    print("\n" + "=" * 120)
    print("🎯 ANÁLISIS DE FIABILIDAD PARA APUESTAS - CORNERS")
    print("=" * 120)
    
    # TOP EQUIPOS FIABLES
    print(f"\n⭐ TOP {top_n} EQUIPOS MÁS FIABLES")
    print("-" * 120)
    
    top_fiables = df_analisis.head(top_n)
    
    for idx, row in top_fiables.iterrows():
        print(f"\n{row['Equipo']:25s} | {row['Nivel']:20s} | Score: {row['Score_Fiabilidad']:.1f}")
        print(f"  📊 Media: {row['Media_CK']:.1f} | Mediana: {row['Mediana_CK']:.1f} | CV: {row['CV_%']:.1f}%")
        print(f"  ✅ {row['Pct_Cerca_Media']:.1f}% cerca de media | IQR: {row['IQR']:.1f}")
        print(f"  ⚠️ Cambios bruscos: {row['Cambios_Bruscos_%']:.1f}% | Outliers: {row['Pct_Outliers']:.1f}%")
        print(f"  📈 Rango: {row['Min']}-{row['Max']} ({row['Rango']} corners)")
    
    # TOP EQUIPOS NO FIABLES
    print(f"\n\n⛔ TOP {top_n} EQUIPOS MENOS FIABLES")
    print("-" * 120)
    
    top_no_fiables = df_analisis.tail(top_n)
    
    for idx, row in top_no_fiables.iterrows():
        print(f"\n{row['Equipo']:25s} | {row['Nivel']:20s} | Score: {row['Score_Fiabilidad']:.1f}")
        print(f"  📊 Media: {row['Media_CK']:.1f} | Mediana: {row['Mediana_CK']:.1f} | CV: {row['CV_%']:.1f}%")
        print(f"  ❌ Solo {row['Pct_Cerca_Media']:.1f}% cerca de media | IQR: {row['IQR']:.1f}")
        print(f"  ⚠️ Cambios bruscos: {row['Cambios_Bruscos_%']:.1f}% | Outliers: {row['Pct_Outliers']:.1f}%")
    
    # ESTADÍSTICAS GENERALES
    print(f"\n\n📊 DISTRIBUCIÓN POR NIVEL DE FIABILIDAD")
    print("-" * 120)
    print(df_analisis['Nivel'].value_counts())
    
    print(f"\n📈 ESTADÍSTICAS DE SCORE:")
    print(f"  Media: {df_analisis['Score_Fiabilidad'].mean():.1f}")
    print(f"  Mediana: {df_analisis['Score_Fiabilidad'].median():.1f}")
    print(f"  Score máximo: {df_analisis['Score_Fiabilidad'].max():.1f}")
    print(f"  Score mínimo: {df_analisis['Score_Fiabilidad'].min():.1f}")

def obtener_fiabilidad_partido(local, visitante, df_analisis):
    """
    Evalúa la fiabilidad de un partido específico
    """
    
    datos_local = df_analisis[df_analisis['Equipo'] == local]
    datos_away = df_analisis[df_analisis['Equipo'] == visitante]
    
    if datos_local.empty or datos_away.empty:
        return {
            'fiabilidad': 'DESCONOCIDO',
            'score': 0,
            'mensaje': '⚠️ Datos insuficientes'
        }
    
    score_local = datos_local['Score_Fiabilidad'].values[0]
    score_away = datos_away['Score_Fiabilidad'].values[0]
    score_promedio = (score_local + score_away) / 2
    
    # Clasificación del partido
    if score_promedio >= 65:
        fiabilidad = "MUY ALTA ⭐⭐⭐"
        mensaje = "✅ EXCELENTE PARTIDO PARA APOSTAR"
    elif score_promedio >= 50:
        fiabilidad = "ALTA ✅"
        mensaje = "✅ BUEN PARTIDO PARA APOSTAR"
    elif score_promedio >= 35:
        fiabilidad = "MEDIA 🟡"
        mensaje = "🟡 APOSTAR CON PRECAUCIÓN"
    else:
        fiabilidad = "BAJA ⛔"
        mensaje = "⛔ EVITAR APUESTA"
    
    return {
        'fiabilidad': fiabilidad,
        'score_local': score_local,
        'score_away': score_away,
        'score_promedio': score_promedio,
        'nivel_local': datos_local['Nivel'].values[0],
        'nivel_away': datos_away['Nivel'].values[0],
        'mensaje': mensaje,
        
        # Datos adicionales útiles
        'cv_local': datos_local['CV_%'].values[0],
        'cv_away': datos_away['CV_%'].values[0],
        'consistencia_local': datos_local['Pct_Cerca_Media'].values[0],
        'consistencia_away': datos_away['Pct_Cerca_Media'].values[0]
    }

def calcular_probabilidades_poisson(lambda_pred, rango_inferior=5, rango_superior=5):
    """Calcula probabilidades usando distribución de Poisson"""
    
    valor_central = int(round(lambda_pred))
    valores_analizar = range(
        max(0, valor_central - rango_inferior),
        valor_central + rango_superior + 1
    )
    
    probabilidades_exactas = {}
    for k in valores_analizar:
        prob = poisson.pmf(k, lambda_pred) * 100
        probabilidades_exactas[k] = prob
    
    # ✅ CORRECCIÓN: MISMAS LÍNEAS PARA OVER Y UNDER
    lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    
    probabilidades_over = {}
    for linea in lines:
        prob_over = (1 - poisson.cdf(linea, lambda_pred)) * 100
        probabilidades_over[linea] = prob_over
    
    probabilidades_under = {}
    for linea in lines:  # ✅ CAMBIO: usar la misma lista
        prob_under = poisson.cdf(linea, lambda_pred) * 100
        probabilidades_under[linea] = prob_under
    
    return {
        'exactas': probabilidades_exactas,
        'over': probabilidades_over,
        'under': probabilidades_under
    }

def clasificar_confianza(prob):
    """Clasifica la confianza según probabilidad"""
    if prob >= 66:
        return "ALTA ✅"
    elif prob >= 55:
        return "MEDIA ⚠️"
    else:
        return "BAJA ❌"

'''
def get_dataframes(df, season, round_num, local, away, league=None):
    """Retorna 8 DataFrames filtrados por equipo, venue y liga"""
    
    season_round = (df['season'] == season) & (df['round'] < round_num)
    
    if league is not None:
        season_round = season_round & (df['league'] == league)
    
    def filter_and_split(team_filter):
        filtered = df[season_round & team_filter].copy()
        home = filtered[filtered['venue'] == "Home"]
        away = filtered[filtered['venue'] == "Away"]
        return home, away
    
    local_home, local_away = filter_and_split(df['team'] == local)
    local_opp_home, local_opp_away = filter_and_split(df['opponent'] == local)
    
    away_home, away_away = filter_and_split(df['team'] == away)
    away_opp_home, away_opp_away = filter_and_split(df['opponent'] == away)
    
    return (local_home, local_away, local_opp_home, local_opp_away,
            away_home, away_away, away_opp_home, away_opp_away)

def get_head_2_head(df, local, away, seasons=None, league=None):
    """Obtiene últimos 3 enfrentamientos directos"""
    if seasons is None:
        seasons = []
    
    df_filtered = df[df['season'].isin(seasons)] if seasons else df
    
    if league is not None:
        df_filtered = df_filtered[df_filtered['league'] == league]
    
    local_h2h = df_filtered[(df_filtered['team'] == local) & (df_filtered['opponent'] == away)]
    away_h2h = df_filtered[(df_filtered['team'] == away) & (df_filtered['opponent'] == local)]
    
    if len(local_h2h) < 4:
        return local_h2h.tail(2), away_h2h.tail(2)
    
    return local_h2h.tail(3), away_h2h.tail(3)

def get_average(df, is_team=False, lst_avg=None):
    """Calcula promedios de estadísticas (VERSIÓN COMPLETA)"""
    
    if len(df) == 0:
        if is_team:
            # ✅ Retornar 23 valores (métricas avanzadas)
            return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    if is_team:
        # ===========================
        # ESTADÍSTICAS BÁSICAS (NORMALIZADAS)
        # ===========================
        avg_cross = (df['Performance_Crs'].sum() / len(df)) - lst_avg[3]
        avg_att_3rd = (df['Touches_Att 3rd'].sum() / len(df)) - lst_avg[4]
        avg_sca = (df['SCA Types_SCA'].sum() / len(df)) - lst_avg[2]
        avg_xg = (df['Expected_xG'].sum() / len(df)) - lst_avg[1]
        
        # ✅ VARIANZA DE CORNERS
        var_ck = df['Pass Types_CK'].var() if len(df) > 1 else 0
        avg_ck = (df['Pass Types_CK'].sum() / len(df)) - lst_avg[8]
        
        avg_poss = (df['Poss'].sum() / len(df)) - 50
        avg_gf = (df['GF'].sum() / len(df)) - lst_avg[5]
        avg_ga = (df['GA'].sum() / len(df)) - lst_avg[6]
        
        # ===========================
        # MÉTRICAS OFENSIVAS AVANZADAS
        # ===========================
        total_sh = df['Standard_Sh'].sum()
        sh_accuracy = (df['Standard_SoT'].sum() / total_sh) if total_sh > 0 else 0
        xg_shot = (df['Expected_xG'].sum() / total_sh) if total_sh > 0 else 0
        
        total_touches = df['Touches_Touches'].sum()
        attacking_presence = (df['Touches_Att 3rd'].sum() / total_touches) if total_touches > 0 else 0
        
        total_poss = df['Poss'].sum()
        possession_shot = (total_sh / total_poss) if total_poss > 0 else 0
        
        # ===========================
        # MÉTRICAS DE CREACIÓN
        # ===========================
        total_passes = df['Total_Att'].sum()
        progressive_pass_ratio = (df['PrgP'].sum() / total_passes) if total_passes > 0 else 0
        final_third_involvement = (df['1/3'].sum() / total_passes) if total_passes > 0 else 0
        
        total_sca = df['SCA Types_SCA'].sum()
        assist_sca = (df['Ast'].sum() / total_sca) if total_sca > 0 else 0
        creative_efficiency = (total_sca / total_poss) if total_poss > 0 else 0
        
        # ===========================
        # MÉTRICAS DEFENSIVAS
        # ===========================
        total_tackles = df['Tackles_Tkl'].sum()
        high_press_intensity = (df['Tackles_Att 3rd'].sum() / total_tackles) if total_tackles > 0 else 0
        interception_tackle = (df['Int'].sum() / total_tackles) if total_tackles > 0 else 0
        
        total_defensive_actions = total_tackles + df['Int'].sum()
        clearance_ratio = (df['Clr'].sum() / total_defensive_actions) if total_defensive_actions > 0 else 0
        
        # ===========================
        # MÉTRICAS DE POSESIÓN
        # ===========================
        total_carries = df['Carries_Carries'].sum()
        progressive_carry_ratio = (df['Carries_PrgC'].sum() / total_carries) if total_carries > 0 else 0
        
        total_prog_passes = df['PrgP'].sum()
        carry_pass_balance = (df['Carries_PrgC'].sum() / total_prog_passes) if total_prog_passes > 0 else 0
        
        # ===========================
        # ÍNDICES COMPUESTOS
        # ===========================
        avg_gf_raw = df['GF'].mean()
        avg_xg_raw = df['Expected_xG'].mean()
        avg_sot = df['Standard_SoT'].mean()
        avg_sh = df['Standard_Sh'].mean()
        offensive_index = (avg_gf_raw + avg_xg_raw) * (avg_sot / avg_sh) if avg_sh > 0 else 0
        
        avg_prgp = df['PrgP'].mean()
        avg_prgc = df['Carries_PrgC'].mean()
        avg_poss_raw = df['Poss'].mean()
        transition_index = ((avg_prgp + avg_prgc) / avg_poss_raw) if avg_poss_raw > 0 else 0
        
        # ✅ RETORNAR 23 VALORES
        return (
            avg_ck,           # 0
            var_ck,           # 1 - ✅ NUEVO
            avg_xg,           # 2
            avg_sca,          # 3
            avg_cross,        # 4
            avg_poss,         # 5
            avg_att_3rd,      # 6
            avg_gf,           # 7
            avg_ga,           # 8
            sh_accuracy,      # 9
            xg_shot,          # 10
            attacking_presence,  # 11
            possession_shot,  # 12
            progressive_pass_ratio,  # 13
            final_third_involvement,  # 14
            assist_sca,       # 15
            creative_efficiency,  # 16
            high_press_intensity,  # 17
            interception_tackle,  # 18
            clearance_ratio,  # 19
            progressive_carry_ratio,  # 20
            carry_pass_balance,  # 21
            offensive_index,  # 22
            transition_index  # 23
        )
    
    # ===========================
    # PROMEDIOS DE LIGA (is_team=False)
    # ===========================
    avg_cross = df['Performance_Crs'].mean()
    avg_att_3rd = df['Touches_Att 3rd'].mean()
    avg_sca = df['SCA Types_SCA'].mean()
    avg_xg = df['Expected_xG'].mean()
    var_ck = df['Pass Types_CK'].var() if len(df) > 1 else 0
    avg_ck = df['Pass Types_CK'].mean()
    avg_gf = df['GF'].mean()
    avg_ga = df['GA'].mean()
    avg_sh = df['Standard_Sh'].mean() if 'Standard_Sh' in df.columns else 0
    
    return (
        var_ck,      # 0
        avg_xg,      # 1
        avg_sca,     # 2
        avg_cross,   # 3
        avg_att_3rd, # 4
        avg_gf,      # 5
        avg_ga,      # 6
        avg_sh,      # 7
        avg_ck       # 8
    )

def get_points_from_result(result):
    """Convierte resultado (W/D/L) a puntos"""
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_team_ppp(df, team, season, round_num, league=None):
    """Calcula puntos por partido (PPP) de un equipo"""
    team_matches = df[
        (df['team'] == team) & 
        (df['season'] == season) & 
        (df['round'] < round_num)
    ]
    
    if league is not None:
        team_matches = team_matches[team_matches['league'] == league]
    
    if len(team_matches) == 0:
        return 0.0
    
    total_points = team_matches['result'].apply(get_points_from_result).sum()
    ppp = total_points / len(team_matches)
    
    return ppp

def get_ppp_difference(df, local, away, season, round_num, league=None):
    """Calcula diferencia de PPP entre local y visitante"""
    local_ppp = get_team_ppp(df, local, season, round_num, league)
    away_ppp = get_team_ppp(df, away, season, round_num, league)
    return local_ppp - away_ppp

'''

def predecir_corners(local, visitante, jornada, temporada="2526", league_code="ESP",df_database=pd.DataFrame(),xgb_model="",xgb_model_local="",xgb_model_away="",xgb_model_xg="",xgb_model_xg_local="",xgb_model_xg_away="",xgb_model_gf="",xgb_model_gf_local="",xgb_model_gf_away="",xgb_model_st="",xgb_model_st_local="",xgb_model_st_away="",scaler="",lst_years=[]):
    """
    Predice corners totales con análisis completo para apuestas
    
    Args:
        local: Equipo local
        visitante: Equipo visitante
        jornada: Número de jornada
        temporada: Temporada (formato "2526")
        league_code: Código de liga ("ESP", "GER", "FRA", "ITA", "NED")
    """
    
    print(f"\n{'='*80}")
    print(f"🏟️  {local} vs {visitante}")
    print(f"📅 Temporada {temporada} | Jornada {jornada} | Liga: {league_code}")
    print(f"{'='*80}")
    
    if jornada < 5:
        return {
            "error": "❌ Se necesitan al menos 5 jornadas previas",
            "prediccion": None
        }
    
    try:
        # ===========================
        # EXTRAER FEATURES (igual que antes)
        # ===========================
        
        lst_avg = get_average(
            df_database[
                (df_database['season'] == temporada) & 
                (df_database['round'] < jornada) &
                (df_database['league'] == league_code)
            ],
            is_team=False
        )
        
        (team1_home, team1_away, team1_opp_home, team1_opp_away,
         team2_home, team2_away, team2_opp_home, team2_opp_away) = get_dataframes(
            df_database, temporada, jornada, local, visitante, league=league_code
        )
        
        index = lst_years.index(temporada)
        result = lst_years[:index+1]
        team1_h2h, team2_h2h = get_head_2_head(
            df_database, local, visitante, seasons=result, league=league_code
        )
        
        local_ppp = get_team_ppp(df_database, local, temporada, jornada, league=league_code)
        away_ppp = get_team_ppp(df_database, visitante, temporada, jornada, league=league_code)
        ppp_diff = local_ppp - away_ppp
        
        # ===========================
        # CONSTRUIR DICCIONARIO DE FEATURES (igual que antes)
        # ===========================
        
        def create_line(df, is_form=True, is_team=False, use_advanced=True):
            if is_form:
                df = df[-6:]
            if use_advanced:
                return get_average(df, is_team, lst_avg)
            else:
                result = get_average(df, is_team, lst_avg)
                return result[:9]
        
        dic_features = {}
        
        dic_features['ppp_local'] = (local_ppp,)
        dic_features['ppp_away'] = (away_ppp,)
        dic_features['ppp_difference'] = (ppp_diff,)
        if jornada < 15:
            dic_features['round'] = (1,)
        elif jornada < 15 and jornada > 25:
            dic_features['round'] = (2,)
        else:
            dic_features['round'] = (3,)

        
        dic_features['lst_team1_home_form'] = create_line(team1_home, True, True, use_advanced=True)
        dic_features['lst_team1_home_general'] = create_line(team1_home, False, True, use_advanced=True)
        dic_features['lst_team1_away_form'] = create_line(team1_away, True, True, use_advanced=True)
        dic_features['lst_team1_away_general'] = create_line(team1_away, False, True, use_advanced=True)
        
        dic_features['lst_team2_home_form'] = create_line(team2_home, True, True, use_advanced=True)
        dic_features['lst_team2_home_general'] = create_line(team2_home, False, True, use_advanced=True)
        dic_features['lst_team2_away_form'] = create_line(team2_away, True, True, use_advanced=True)
        dic_features['lst_team2_away_general'] = create_line(team2_away, False, True, use_advanced=True)
        
        dic_features['lst_team1_h2h'] = create_line(team1_h2h, False, True, use_advanced=True)
        dic_features['lst_team2_h2h'] = create_line(team2_h2h, False, True, use_advanced=True)
        
        dic_features['lst_team1_opp_away'] = create_line(team1_opp_away, False, True, use_advanced=False)
        dic_features['lst_team2_opp_home'] = create_line(team2_opp_home, False, True, use_advanced=False)
        
        league_dummies = {
                    'league_ESP': 1 if league_code == 'ESP' else 0,
                    'league_GER': 1 if league_code == 'GER' else 0,
                    'league_FRA': 1 if league_code == 'FRA' else 0,
                    'league_ITA': 1 if league_code == 'ITA' else 0,
                    'league_NED': 1 if league_code == 'NED' else 0,
                    'league_ENG': 1 if league_code == 'ENG' else 0,
                    'league_POR': 1 if league_code == 'POR' else 0,
                    'league_BEL': 1 if league_code == 'BEL' else 0
                }
        
        for key, value in league_dummies.items():
            dic_features[key] = (value,)
        
        # ===========================
        # CONSTRUIR VECTOR DE FEATURES
        # ===========================
        
        lst_base_advanced = [
            "avg_ck", "var_ck", "xg", "sca", "cross", "poss", "att_3rd", "gf", "ga",
            "sh_accuracy", "xg_shot", "attacking_presence", "possession_shot",
            "progressive_pass_ratio", "final_third_involvement", "assist_sca", "creative_efficiency",
            "high_press_intensity", "interception_tackle", "clearance_ratio",
            "progressive_carry_ratio", "carry_pass_balance", "offensive_index", "transition_index"
        ]
        
        lst_base_original = [
            "var_ck", "xg", "sca", "cross", "poss", "att_3rd", "gf", "ga", "avg_ck"
        ]
        
        lst_features_values = []
        lst_features_names = []
        
        for key in dic_features:
            lst_features_values.extend(list(dic_features[key]))
            
            if key in ['ppp_local', 'ppp_away', 'ppp_difference','round']:
                lst_features_names.append(key)
            elif key.startswith('league_'):
                lst_features_names.append(key)
            elif key in ['lst_team1_opp_away', 'lst_team2_opp_home']:
                lst_features_names.extend([f"{key}_{col}" for col in lst_base_original])
            else:
                lst_features_names.extend([f"{key}_{col}" for col in lst_base_advanced])
        
        df_input = pd.DataFrame([lst_features_values], columns=lst_features_names)
        
        expected_features = scaler.feature_names_in_
        
        if len(df_input.columns) != len(expected_features):
            print(f"\n⚠️ ERROR: Número de features no coincide")
            print(f"   Esperadas: {len(expected_features)}")
            print(f"   Recibidas: {len(df_input.columns)}")
            return {"error": "Desajuste de features", "prediccion": None}
        
        df_input = df_input[expected_features]
        
        X_input_scaled = pd.DataFrame(
            scaler.transform(df_input), 
            columns=df_input.columns
        )
        
        # ===========================
        # PREDICCIÓN
        # ===========================
        
        prediccion = xgb_model.predict(X_input_scaled)[0]
        prediccion_local = xgb_model_local.predict(X_input_scaled)[0]
        prediccion_away = xgb_model_away.predict(X_input_scaled)[0]
        prediccion_xg = xgb_model_xg.predict(X_input_scaled)[0]
        prediccion_xg_local = xgb_model_xg_local.predict(X_input_scaled)[0]
        prediccion_xg_away = xgb_model_xg_away.predict(X_input_scaled)[0]
        prediccion_gf = xgb_model_gf.predict(X_input_scaled)[0]
        prediccion_gf_local = xgb_model_gf_local.predict(X_input_scaled)[0]
        prediccion_gf_away = xgb_model_gf_away.predict(X_input_scaled)[0]
        prediccion_st = xgb_model_st.predict(X_input_scaled)[0]
        prediccion_st_local = xgb_model_st_local.predict(X_input_scaled)[0]
        prediccion_st_away = xgb_model_st_away.predict(X_input_scaled)[0]
        
        # ===========================
        # ✅ ANÁLISIS PROBABILÍSTICO CON POISSON
        # ===========================
        
        analisis = calcular_probabilidades_poisson(prediccion, rango_inferior=5, rango_superior=5)
        
        # ===========================
        # ESTADÍSTICAS DETALLADAS
        # ===========================
        def get_stat(df, col, form=False):
            if len(df) == 0: return 0
            data = df.tail(6) if form else df
            if col not in data.columns: return 0
            return data[col].mean()

        # --- 1. EQUIPO LOCAL (TEAM 1) ---
        
        # 1.1 Cuando juega en CASA (Home)
        local_ck_home = get_stat(team1_home, 'Pass Types_CK')
        local_xg_home = get_stat(team1_home, 'Expected_xG')
        local_gf_home = get_stat(team1_home, 'GF')
        local_st_home = get_stat(team1_home, 'Standard_SoT') # Nuevo

        print(team1_home)
        print(team2_away)
        
        local_ck_home_form = get_stat(team1_home, 'Pass Types_CK', form=True)
        local_xg_home_form = get_stat(team1_home, 'Expected_xG', form=True)
        local_gf_home_form = get_stat(team1_home, 'GF', form=True)
        local_st_home_form = get_stat(team1_home, 'Standard_SoT', form=True) # Nuevo

        local_ck_received_home = get_stat(team1_opp_home, 'Pass Types_CK')
        local_gf_received_home = get_stat(team1_opp_home, 'GF')
        local_xg_received_home = get_stat(team1_opp_home, 'Expected_xG')
        local_st_received_home = get_stat(team1_opp_home, 'Standard_SoT') # Nuevo
        
        local_ck_received_home_form = get_stat(team1_opp_home, 'Pass Types_CK', form=True)
        local_gf_received_home_form = get_stat(team1_opp_home, 'GF', form=True)
        local_xg_received_home_form = get_stat(team1_opp_home, 'Expected_xG', form=True)
        local_st_received_home_form = get_stat(team1_opp_home, 'Standard_SoT', form=True) # Nuevo

        # 1.2 Cuando juega FUERA (Away) - NUEVO
        local_ck_away = get_stat(team1_away, 'Pass Types_CK')
        local_xg_away = get_stat(team1_away, 'Expected_xG')
        local_gf_away = get_stat(team1_away, 'GF')
        local_st_away = get_stat(team1_away, 'Standard_SoT') # Nuevo
        
        local_ck_away_form = get_stat(team1_away, 'Pass Types_CK', form=True)
        local_xg_away_form = get_stat(team1_away, 'Expected_xG', form=True)
        local_gf_away_form = get_stat(team1_away, 'GF', form=True)
        local_st_away_form = get_stat(team1_away, 'Standard_SoT', form=True) # Nuevo

        local_ck_received_away = get_stat(team1_opp_away, 'Pass Types_CK')
        local_gf_received_away = get_stat(team1_opp_away, 'GF')
        local_xg_received_away = get_stat(team1_opp_away, 'Expected_xG')
        local_st_received_away = get_stat(team1_opp_away, 'Standard_SoT') # Nuevo

        local_ck_received_away_form = get_stat(team1_opp_away, 'Pass Types_CK', form=True)
        local_gf_received_away_form = get_stat(team1_opp_away, 'GF', form=True)
        local_xg_received_away_form = get_stat(team1_opp_away, 'Expected_xG', form=True)
        local_st_received_away_form = get_stat(team1_opp_away, 'Standard_SoT', form=True) # Nuevo


        # --- 2. EQUIPO VISITANTE (TEAM 2) ---

        # 2.1 Cuando juega FUERA (Away)
        away_ck_away = get_stat(team2_away, 'Pass Types_CK')
        away_xg_away = get_stat(team2_away, 'Expected_xG')
        away_gf_away = get_stat(team2_away, 'GF')
        away_st_away = get_stat(team2_away, 'Standard_SoT') # Nuevo

        away_ck_away_form = get_stat(team2_away, 'Pass Types_CK', form=True)
        away_xg_away_form = get_stat(team2_away, 'Expected_xG', form=True)
        away_gf_away_form = get_stat(team2_away, 'GF', form=True)
        away_st_away_form = get_stat(team2_away, 'Standard_SoT', form=True) # Nuevo

        away_ck_received_away = get_stat(team2_opp_away, 'Pass Types_CK')
        away_gf_received_away = get_stat(team2_opp_away, 'GF')
        away_xg_received_away = get_stat(team2_opp_away, 'Expected_xG')
        away_st_received_away = get_stat(team2_opp_away, 'Standard_SoT') # Nuevo

        away_ck_received_away_form = get_stat(team2_opp_away, 'Pass Types_CK', form=True)
        away_gf_received_away_form = get_stat(team2_opp_away, 'GF', form=True)
        away_xg_received_away_form = get_stat(team2_opp_away, 'Expected_xG', form=True)
        away_st_received_away_form = get_stat(team2_opp_away, 'Standard_SoT', form=True) # Nuevo

        # 2.2 Cuando juega en CASA (Home) - NUEVO
        away_ck_home = get_stat(team2_home, 'Pass Types_CK')
        away_xg_home = get_stat(team2_home, 'Expected_xG')
        away_gf_home = get_stat(team2_home, 'GF')
        away_st_home = get_stat(team2_home, 'Standard_SoT') # Nuevo

        away_ck_home_form = get_stat(team2_home, 'Pass Types_CK', form=True)
        away_xg_home_form = get_stat(team2_home, 'Expected_xG', form=True)
        away_gf_home_form = get_stat(team2_home, 'GF', form=True)
        away_st_home_form = get_stat(team2_home, 'Standard_SoT', form=True) # Nuevo

        away_ck_received_home = get_stat(team2_opp_home, 'Pass Types_CK')
        away_gf_received_home = get_stat(team2_opp_home, 'GF')
        away_xg_received_home = get_stat(team2_opp_home, 'Expected_xG')
        away_st_received_home = get_stat(team2_opp_home, 'Standard_SoT') # Nuevo

        away_ck_received_home_form = get_stat(team2_opp_home, 'Pass Types_CK', form=True)
        away_gf_received_home_form = get_stat(team2_opp_home, 'GF', form=True)
        away_xg_received_home_form = get_stat(team2_opp_home, 'Expected_xG', form=True)
        away_st_received_home_form = get_stat(team2_opp_home, 'Standard_SoT', form=True) # Nuevo


        # --- TOTALES ESPERADOS (Contexto del partido actual: Local en Casa vs Visitante Fuera) ---
        partido_ck_esperado = local_ck_home + away_ck_away
        partido_gf_esperado = local_gf_home + away_gf_away
        partido_xg_esperado = local_xg_home + away_xg_away
        partido_st_esperado = local_st_home + away_st_away # Nuevo

        partido_ck_esperado_form = local_ck_home_form + away_ck_away_form 
        partido_gf_esperado_form = local_gf_home_form + away_gf_away_form
        partido_xg_esperado_form = local_xg_home_form + away_xg_away_form
        partido_st_esperado_form = local_st_home_form + away_st_away_form # Nuevo
        
        # --- H2H ---
        h2h_ck_local = get_stat(team1_h2h, 'Pass Types_CK')
        h2h_ck_away = get_stat(team2_h2h, 'Pass Types_CK')
        h2h_ck_total = h2h_ck_local + h2h_ck_away

        h2h_gf_local = get_stat(team1_h2h, 'GF')
        h2h_gf_away = get_stat(team2_h2h, 'GF')
        h2h_gf_total = h2h_gf_local + h2h_gf_away

        h2h_xg_local = get_stat(team1_h2h, 'Expected_xG')
        h2h_xg_away = get_stat(team2_h2h, 'Expected_xG')
        h2h_xg_total = h2h_xg_local + h2h_xg_away
        
        h2h_st_local = get_stat(team1_h2h, 'Standard_SoT') # Nuevo
        h2h_st_away = get_stat(team2_h2h, 'Standard_SoT') # Nuevo
        h2h_st_total = h2h_st_local + h2h_st_away # Nuevo
        
        # --- H2H DATAFRAME ---
        h2h_matches = []
        
        # Asegurar que trabajamos con copias limpias
        t1_h2h = team1_h2h.copy()
        t2_h2h = team2_h2h.copy()
        
        # Iterar sobre los partidos del equipo local (team1)
        for idx, row1 in t1_h2h.iterrows():
            # Buscar el partido correspondiente en el df del visitante (team2)
            # Coincidencia por temporada y jornada
            row2_match = t2_h2h[
                (t2_h2h['season'] == row1['season']) & 
                (t2_h2h['round'] == row1['round'])
            ]
            
            if not row2_match.empty:
                row2 = row2_match.iloc[0]
                
                # Determinar quién jugó en casa en ese partido histórico
                if row1['venue'] == 'Home':
                    match_home_team = local
                    match_away_team = visitante
                else:
                    match_home_team = visitante
                    match_away_team = local
                
                match_data = {
                    'season': row1['season'],
                    'round': int(row1['round']) if pd.notna(row1['round']) else 0,
                    'match_home_team': match_home_team,
                    'match_away_team': match_away_team,
                    
                    # Datos del equipo que es LOCAL en la predicción actual (team1)
                    'local_team_stats': {
                        'team': local,
                        'venue': row1['venue'],
                        'goals': int(row1['GF']),
                        'corners': int(row1['Pass Types_CK']),
                        'xg': float(row1['Expected_xG']),
                        'sot': int(row1['Standard_SoT']) if 'Standard_SoT' in row1 else 0
                    },
                    
                    # Datos del equipo que es VISITANTE en la predicción actual (team2)
                    'away_team_stats': {
                        'team': visitante,
                        'venue': row2['venue'],
                        'goals': int(row2['GF']),
                        'corners': int(row2['Pass Types_CK']),
                        'xg': float(row2['Expected_xG']),
                        'sot': int(row2['Standard_SoT']) if 'Standard_SoT' in row2 else 0
                    }
                }
                h2h_matches.append(match_data)

        print(h2h_matches)

        
        # ===========================
        # ✅ MOSTRAR PROBABILIDADES EXACTAS
        # ===========================
        
        valor_mas_probable = max(analisis['exactas'].items(), key=lambda x: x[1])
        
        print(f"\n📈 PROBABILIDADES EXACTAS (Poisson):")
        for k in sorted(analisis['exactas'].keys()):
            prob = analisis['exactas'][k]
            bar = '█' * int(prob / 2)
            marca = ' ⭐' if k == valor_mas_probable[0] else ''
            print(f"   {k:2d} corners: {prob:5.2f}% {bar}{marca}")
        
        print(f"\n✅ Valor más probable: {valor_mas_probable[0]} corners ({valor_mas_probable[1]:.2f}%)")
        
        # ✅ RANGO DE 80% CONFIANZA
        probs_sorted = sorted(analisis['exactas'].items(), key=lambda x: x[1], reverse=True)
        cumsum = 0
        rango_80 = []
        for val, prob in probs_sorted:
            cumsum += prob
            rango_80.append(val)
            if cumsum >= 80:
                break
        
        print(f"📊 Rango 80% confianza: {min(rango_80)}-{max(rango_80)} corners")
        
        # ===========================
        # ✅ MOSTRAR OVER/UNDER CON CUOTAS IMPLÍCITAS
        # ===========================
        
        print(f"\n🎯 ANÁLISIS OVER/UNDER:")
        print(f"{'Línea':<10} {'Prob Over':<12} {'Cuota Impl':<12} {'Confianza':<15} {'Prob Under':<12} {'Cuota Impl':<12}")
        print("-" * 85)
        
        for linea in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
            prob_over = analisis['over'][linea]
            prob_under = analisis['under'][linea]
            
            # Cuotas implícitas (inverso de probabilidad en decimal)
            cuota_impl_over = 100 / prob_over if prob_over > 0 else 999
            cuota_impl_under = 100 / prob_under if prob_under > 0 else 999
            
            conf_over = clasificar_confianza(prob_over)
            
            print(f"O/U {linea:<5} {prob_over:6.2f}%     @{cuota_impl_over:5.2f}      {conf_over:<15} {prob_under:6.2f}%     @{cuota_impl_under:5.2f}")
        
        # ===========================
        # ✅ RECOMENDACIONES CON CUOTAS
        # ===========================
        
        print(f"\n💡 RECOMENDACIONES DE APUESTA:")
        
        mejores_over = [(l, p) for l, p in analisis['over'].items() if p >= 55]
        mejores_under = [(l, p) for l, p in analisis['under'].items() if p >= 55]
        
        if mejores_over:
            print(f"\n✅ OVER con confianza MEDIA/ALTA:")
            for linea, prob in sorted(mejores_over, key=lambda x: x[1], reverse=True):
                cuota_impl = 100 / prob
                conf = clasificar_confianza(prob)
                print(f"   • Over {linea}: {prob:.2f}% (Cuota justa: @{cuota_impl:.2f}) - {conf}")
        
        if mejores_under:
            print(f"\n✅ UNDER con confianza MEDIA/ALTA:")
            for linea, prob in sorted(mejores_under, key=lambda x: x[1], reverse=True):
                cuota_impl = 100 / prob
                conf = clasificar_confianza(prob)
                print(f"   • Under {linea}: {prob:.2f}% (Cuota justa: @{cuota_impl:.2f}) - {conf}")
        
        if not mejores_over and not mejores_under:
            print(f"   ⚠️ No hay apuestas con confianza MEDIA o superior")
        
        # ===========================
        # ✅ ANÁLISIS DE RIESGO
        # ===========================
        
        df_varianza_temp = analizar_fiabilidad_equipos(df_database, temporada=temporada, min_partidos=3)
        riesgo = obtener_fiabilidad_partido(local, visitante, df_varianza_temp)

        
        # ===========================
        # RETORNAR DICCIONARIO COMPLETO
        # ===========================
        
        return {
            "prediccion": round(prediccion, 2),
            "prediccion_local": round(prediccion_local, 2),
            "prediccion_away": round(prediccion_away, 2),
            "prediccion_xg": round(prediccion_xg, 2),
            "prediccion_xg_local": round(prediccion_xg_local, 2),
            "prediccion_xg_away": round(prediccion_xg_away, 2),
            "prediccion_gf": round(prediccion_gf, 2),
            "prediccion_gf_local": round(prediccion_gf_local, 2),
            "prediccion_gf_away": round(prediccion_gf_away, 2),
            "prediccion_st": round(prediccion_st, 2),
            "prediccion_st_local": round(prediccion_st_local, 2),
            "prediccion_st_away": round(prediccion_st_away, 2),
            "local": local,
            "visitante": visitante,
            "ppp_local": local_ppp,
            "ppp_away": away_ppp,
            "ppp_diff": ppp_diff,
            "h2h_matches": h2h_matches, # Nuevo
            "riesgo": riesgo,
            "stats_ck": {
                # Local Team Stats
                "local_ck_home": local_ck_home,
                "local_ck_away": local_ck_away, # Nuevo
                "local_ck_received_home": local_ck_received_home,
                "local_ck_received_away": local_ck_received_away, # Nuevo
                
                "local_ck_home_form": local_ck_home_form,
                "local_ck_away_form": local_ck_away_form, # Nuevo
                "local_ck_received_home_form": local_ck_received_home_form,
                "local_ck_received_away_form": local_ck_received_away_form, # Nuevo

                # Away Team Stats
                "away_ck_home": away_ck_home, # Nuevo
                "away_ck_away": away_ck_away,
                "away_ck_received_home": away_ck_received_home, # Nuevo
                "away_ck_received_away": away_ck_received_away,

                "away_ck_home_form": away_ck_home_form, # Nuevo
                "away_ck_away_form": away_ck_away_form,
                "away_ck_received_home_form": away_ck_received_home_form, # Nuevo
                "away_ck_received_away_form": away_ck_received_away_form,

                # Totals
                "h2h_ck_total": h2h_ck_total,
                "partido_ck_esperado": partido_ck_esperado,
                "partido_ck_esperado_form": partido_ck_esperado_form
            },
            "stats_gf": {
                # Local Team Stats
                "local_gf_home": local_gf_home,
                "local_gf_away": local_gf_away, # Nuevo
                "local_gf_received_home": local_gf_received_home,
                "local_gf_received_away": local_gf_received_away, # Nuevo

                "local_gf_home_form": local_gf_home_form,
                "local_gf_away_form": local_gf_away_form, # Nuevo
                "local_gf_received_home_form": local_gf_received_home_form,
                "local_gf_received_away_form": local_gf_received_away_form, # Nuevo

                # Away Team Stats
                "away_gf_home": away_gf_home, # Nuevo
                "away_gf_away": away_gf_away,
                "away_gf_received_home": away_gf_received_home, # Nuevo
                "away_gf_received_away": away_gf_received_away,

                "away_gf_home_form": away_gf_home_form, # Nuevo
                "away_gf_away_form": away_gf_away_form,
                "away_gf_received_home_form": away_gf_received_home_form, # Nuevo
                "away_gf_received_away_form": away_gf_received_away_form,

                # Totals
                "h2h_gf_total": h2h_gf_total,
                "partido_gf_esperado": partido_gf_esperado,
                "partido_gf_esperado_form": partido_gf_esperado_form,
            },
            "stats_xg": {
                # Local Team Stats
                "local_xg_home": local_xg_home,
                "local_xg_away": local_xg_away, # Nuevo
                "local_xg_received_home": local_xg_received_home,
                "local_xg_received_away": local_xg_received_away, # Nuevo

                "local_xg_home_form": local_xg_home_form,
                "local_xg_away_form": local_xg_away_form, # Nuevo
                "local_xg_received_home_form": local_xg_received_home_form,
                "local_xg_received_away_form": local_xg_received_away_form, # Nuevo

                # Away Team Stats
                "away_xg_home": away_xg_home, # Nuevo
                "away_xg_away": away_xg_away,
                "away_xg_received_home": away_xg_received_home, # Nuevo
                "away_xg_received_away": away_xg_received_away,

                "away_xg_home_form": away_xg_home_form, # Nuevo
                "away_xg_away_form": away_xg_away_form,
                "away_xg_received_home_form": away_xg_received_home_form, # Nuevo
                "away_xg_received_away_form": away_xg_received_away_form,

                # Totals
                "h2h_xg_total": h2h_xg_total,
                "partido_xg_esperado": partido_xg_esperado,
                "partido_xg_esperado_form": partido_xg_esperado_form
            },
            "stats_st": { # Nuevo bloque
                # Local Team Stats
                "local_st_home": local_st_home,
                "local_st_away": local_st_away,
                "local_st_received_home": local_st_received_home,
                "local_st_received_away": local_st_received_away,

                "local_st_home_form": local_st_home_form,
                "local_st_away_form": local_st_away_form,
                "local_st_received_home_form": local_st_received_home_form,
                "local_st_received_away_form": local_st_received_away_form,

                # Away Team Stats
                "away_st_home": away_st_home,
                "away_st_away": away_st_away,
                "away_st_received_home": away_st_received_home,
                "away_st_received_away": away_st_received_away,

                "away_st_home_form": away_st_home_form,
                "away_st_away_form": away_st_away_form,
                "away_st_received_home_form": away_st_received_home_form,
                "away_st_received_away_form": away_st_received_away_form,

                # Totals
                "h2h_st_total": h2h_st_total,
                "partido_st_esperado": partido_st_esperado,
                "partido_st_esperado_form": partido_st_esperado_form,
            },
            "probabilidades_exactas": analisis['exactas'],
            "probabilidades_over": analisis['over'],
            "probabilidades_under": analisis['under'],
            "valor_mas_probable": valor_mas_probable[0],
            "prob_mas_probable": valor_mas_probable[1],
            "rango_80": (min(rango_80), max(rango_80))
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "prediccion": None}

def predecir_partidos_batch(partidos, jornada, temporada="2526", league_code="ESP", export_csv=True, filename=None,df_database=pd.DataFrame(),xgb_model="",scaler="",lst_years=[]):
    """
    Predice corners para múltiples partidos y exporta resultados a CSV
    
    Args:
        partidos: Lista de tuplas [(local1, visitante1), (local2, visitante2), ...]
        jornada: Número de jornada
        temporada: Temporada (formato "2526")
        league_code: Código de liga ("ESP", "GER", "FRA", "ITA", "NED")
        export_csv: Si True, exporta a CSV
        filename: Nombre del archivo CSV (opcional)
    
    Returns:
        DataFrame con todos los resultados
    """
    
    resultados = []
    
    print("\n" + "=" * 120)
    print(f"🎯 PROCESANDO {len(partidos)} PARTIDOS - {league_code} | J{jornada} | Temporada {temporada}")
    print("=" * 120)
    
    for idx, (local, visitante) in enumerate(partidos, 1):
        print(f"\n[{idx}/{len(partidos)}] Procesando: {local} vs {visitante}...")
        
        resultado = predecir_corners(
            local=local,
            visitante=visitante,
            jornada=jornada,
            temporada=temporada,
            league_code=league_code,
            df_database=df_database,
            xgb_model=xgb_model,
            scaler=scaler,
            lst_years=lst_years)
        
        
        if resultado.get("error"):
            print(f"   ❌ Error: {resultado['error']}")
            continue
        
        # ===========================
        # CONSTRUIR FILA DE DATOS
        # ===========================
        
        fila = {
            'Partido': f"{local} vs {visitante}",
            'Local': local,
            'Visitante': visitante,
            'Liga': league_code,
            'Jornada': jornada,
            'Temporada': temporada,
            
            # Predicción
            'Prediccion': resultado['prediccion'],
            'Valor_Mas_Probable': resultado['valor_mas_probable'],
            'Prob_Valor_Mas_Probable_%': round(resultado['prob_mas_probable'], 2),
            'Rango_80%_Min': resultado['rango_80'][0],
            'Rango_80%_Max': resultado['rango_80'][1],
            
            # PPP
            'PPP_Local': round(resultado['ppp_local'], 2),
            'PPP_Away': round(resultado['ppp_away'], 2),
            'PPP_Diferencia': round(resultado['ppp_diff'], 2),
            
            # Estadísticas históricas
            'CK_Local_Casa': round(resultado['stats']['local_ck'], 1),
            'CK_Away_Fuera': round(resultado['stats']['away_ck'], 1),
            'CK_Local_Recibidos': round(resultado['stats']['local_ck_received'], 1),
            'CK_Away_Recibidos': round(resultado['stats']['away_ck_received'], 1),
            'CK_Esperado_Suma': round(resultado['stats']['partido_esperado'], 1),
            'CK_H2H_Total': round(resultado['stats']['h2h_total'], 1) if resultado['stats']['h2h_total'] > 0 else 'N/A',
            
            # Riesgo
            'Fiabilidad_Partido': resultado['riesgo']['fiabilidad'],
            'Score_Fiabilidad': round(resultado['riesgo']['score_promedio'], 1),
            'Nivel_Local': resultado['riesgo']['nivel_local'],
            'Nivel_Away': resultado['riesgo']['nivel_away'],
            'CV_Local_%': round(resultado['riesgo']['cv_local'], 1),
            'CV_Away_%': round(resultado['riesgo']['cv_away'], 1),
        }
        
        # ===========================
        # OVER 6.5 a 10.5
        # ===========================
        for linea in [6.5, 7.5, 8.5, 9.5, 10.5]:
            prob = resultado['probabilidades_over'].get(linea, 0)
            cuota_impl = round(100 / prob, 2) if prob > 0 else 999
            conf = clasificar_confianza(prob)
            
            fila[f'Over_{linea}_Prob_%'] = round(prob, 2)
            fila[f'Over_{linea}_Cuota'] = cuota_impl
            fila[f'Over_{linea}_Confianza'] = conf
        
        # ===========================
        # UNDER 12.5 a 9.5
        # ===========================
        for linea in [12.5, 11.5, 10.5, 9.5]:
            prob = resultado['probabilidades_under'].get(linea, 0)
            cuota_impl = round(100 / prob, 2) if prob > 0 else 999
            conf = clasificar_confianza(prob)
            
            fila[f'Under_{linea}_Prob_%'] = round(prob, 2)
            fila[f'Under_{linea}_Cuota'] = cuota_impl
            fila[f'Under_{linea}_Confianza'] = conf
        
        # ===========================
        # RECOMENDACIONES
        # ===========================
        
        mejores_over = [(l, p) for l, p in resultado['probabilidades_over'].items() if p >= 55]
        mejores_under = [(l, p) for l, p in resultado['probabilidades_under'].items() if p >= 55]
        
        if resultado['riesgo']['score_promedio'] < 35:
            fila['Recomendacion'] = "⛔ EVITAR - Baja fiabilidad"
            fila['Es_Apostable'] = "NO"
        elif not mejores_over and not mejores_under:
            fila['Recomendacion'] = "⚠️ NO RECOMENDADO - Sin confianza suficiente"
            fila['Es_Apostable'] = "NO"
        else:
            recomendaciones = []
            
            if mejores_over:
                mejor_over = max(mejores_over, key=lambda x: x[1])
                cuota_over = round(100 / mejor_over[1], 2)
                recomendaciones.append(f"Over {mejor_over[0]} ({mejor_over[1]:.1f}% @{cuota_over})")
            
            if mejores_under:
                mejor_under = max(mejores_under, key=lambda x: x[1])
                cuota_under = round(100 / mejor_under[1], 2)
                recomendaciones.append(f"Under {mejor_under[0]} ({mejor_under[1]:.1f}% @{cuota_under})")
            
            fila['Recomendacion'] = " | ".join(recomendaciones)
            
            if resultado['riesgo']['score_promedio'] >= 65:
                fila['Es_Apostable'] = "SÍ ⭐⭐⭐"
            elif resultado['riesgo']['score_promedio'] >= 50:
                fila['Es_Apostable'] = "SÍ ✅"
            else:
                fila['Es_Apostable'] = "PRECAUCIÓN 🟡"
        
        fila['Mensaje_Riesgo'] = resultado['riesgo']['mensaje']
        
        resultados.append(fila)
        print(f"   ✅ Completado")
    
    # ===========================
    # CREAR DATAFRAME
    # ===========================
    
    df_resultados = pd.DataFrame(resultados)
    
    print("\n" + "=" * 120)
    print(f"✅ PROCESAMIENTO COMPLETADO: {len(df_resultados)} partidos analizados")
    print("=" * 120)
    
    # ===========================
    # EXPORTAR A CSV
    # ===========================
    
    if export_csv and len(df_resultados) > 0:
        if filename is None:
            filename = f"predicciones_{league_code}_J{jornada}_{temporada}.csv"
        
        df_resultados.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n💾 Resultados exportados a: {filename}")
    
    # ===========================
    # RESUMEN
    # ===========================
    
    print(f"\n📊 RESUMEN DE APUESTAS:")
    print(f"   Partidos apostables: {len(df_resultados[df_resultados['Es_Apostable'].str.contains('SÍ')])} / {len(df_resultados)}")
    print(f"   Partidos ALTA confianza (⭐⭐⭐): {len(df_resultados[df_resultados['Es_Apostable'] == 'SÍ ⭐⭐⭐'])}")
    print(f"   Partidos MEDIA confianza (✅): {len(df_resultados[df_resultados['Es_Apostable'] == 'SÍ ✅'])}")
    print(f"   Partidos a evitar (⛔): {len(df_resultados[df_resultados['Es_Apostable'] == 'NO'])}")
    
    return df_resultados

def mostrar_resumen_batch(df_resultados):
    """Muestra resumen visual de los resultados"""
    
    print("\n" + "=" * 120)
    print("🎯 MEJORES OPORTUNIDADES DE APUESTA")
    print("=" * 120)
    
    # Filtrar solo apostables
    df_apostables = df_resultados[df_resultados['Es_Apostable'].str.contains('SÍ')].copy()
    
    if len(df_apostables) == 0:
        print("\n⚠️ No se encontraron partidos con oportunidades de apuesta")
        return
    
    # Ordenar por score de fiabilidad
    df_apostables = df_apostables.sort_values('Score_Fiabilidad', ascending=False)
    
    for idx, row in df_apostables.iterrows():
        print(f"\n{'='*120}")
        print(f"🏟️  {row['Partido']}")
        print(f"{'='*120}")
        print(f"📊 Predicción: {row['Prediccion']:.2f} corners | Valor más probable: {row['Valor_Mas_Probable']} ({row['Prob_Valor_Mas_Probable_%']:.1f}%)")
        print(f"📈 Histórico: Local {row['CK_Local_Casa']:.1f} CK | Away {row['CK_Away_Fuera']:.1f} CK | H2H: {row['CK_H2H_Total']}")
        print(f"🎲 Fiabilidad: {row['Fiabilidad_Partido']} (Score: {row['Score_Fiabilidad']:.1f}/100)")
        print(f"💡 {row['Recomendacion']}")
        
        # Mostrar líneas con alta probabilidad
        print(f"\n   📌 Líneas destacadas:")
        for linea in [7.5, 8.5, 9.5, 10.5]:
            over_prob = row.get(f'Over_{linea}_Prob_%', 0)
            under_prob = row.get(f'Under_{linea}_Prob_%', 0)
            
            if over_prob >= 55:
                cuota = row.get(f'Over_{linea}_Cuota', 0)
                conf = row.get(f'Over_{linea}_Confianza', '')
                print(f"   • Over {linea}: {over_prob:.1f}% @{cuota:.2f} - {conf}")
            
            if under_prob >= 55:
                cuota = row.get(f'Under_{linea}_Cuota', 0)
                conf = row.get(f'Under_{linea}_Confianza', '')
                print(f"   • Under {linea}: {under_prob:.1f}% @{cuota:.2f} - {conf}")




class USE_MODEL():
    def __init__(self):
        self.load_models()
        self.load_data()
        self.init_variables()

    def load_models(self):
        """Cargar modelos desde GitHub usando raw URLs"""
        
        print("📦 Cargando modelos desde GitHub...")
        
        # URLs de descarga directa (raw.githubusercontent.com)
        base_url = "https://raw.githubusercontent.com/danielsaed/futbol_corners_forecast/refs/heads/main/models"
        #model_url = f"{base_url}/xgboost_corners_v4_retrain.pkl"
        models_map = {
            "xgb_model": "xgboost_corners_v4_retrain_PRODUCTION.pkl",
            "xgb_model_local": "xgboost_corners_v4_retrain_local_PRODUCTION.pkl",
            "xgb_model_away": "xgboost_corners_v4_retrain_away_PRODUCTION.pkl",
            "xgb_model_xg": "xgboost_corners_v4_retrain_eg_total_PRODUCTION.pkl",
            "xgb_model_xg_local": "xgboost_corners_v4_retrain_eg_local_PRODUCTION.pkl",
            "xgb_model_xg_away": "xgboost_corners_v4_retrain_eg_away_PRODUCTION.pkl",
            "xgb_model_gf": "xgboost_corners_v4_retrain_gol_total_PRODUCTION.pkl",
            "xgb_model_gf_local": "xgboost_corners_v4_retrain_gol_local_PRODUCTION.pkl",
            "xgb_model_gf_away": "xgboost_corners_v4_retrain_gol_away_PRODUCTION.pkl",
            "xgb_model_st": "xgboost_corners_v4_retrain_st_total_PRODUCTION.pkl",
            "xgb_model_st_local": "xgboost_corners_v4_retrain_st_local_PRODUCTION.pkl",
            "xgb_model_st_away": "xgboost_corners_v4_retrain_st_away_PRODUCTION.pkl",
            "scaler": "scaler_corners_v4_retrain_PRODUCTION.pkl"
        }
        
        try:
            for attr_name, filename in models_map.items():
                url = f"{base_url}/{filename}"
                print(f"📥 Descargando {attr_name} desde: {url}")
                
                # 1. Descargar
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # 2. Guardar temporalmente
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                
                # 3. Cargar con joblib y asignar al atributo de la clase
                try:
                    loaded_obj = joblib.load(tmp_path)
                    setattr(self, attr_name, loaded_obj)
                finally:
                    # 4. Limpiar archivo temporal
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            print("✅ Todos los modelos cargados correctamente desde GitHub")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"❌ Error descargando modelos: {str(e)}")
        except Exception as e:
            raise Exception(f"❌ Error cargando modelos: {str(e)}")

    def load_data(self):
        """Cargar datos desde GitHub"""
        
        print("📂 Cargando datos desde GitHub...")
        
        base_url = "https://raw.githubusercontent.com/danielsaed/futbol_corners_forecast/refs/heads/main/dataset/cleaned"
        historic_url = f"{base_url}/dataset_cleaned.csv"
        current_url = f"{base_url}/dataset_cleaned_current_year.csv"
        
        try:
            # Cargar dataset histórico
            print(f"📥 Descargando dataset histórico...")
            self.df_dataset_historic = pd.read_csv(historic_url)
            print(f"✅ Dataset histórico cargado: {len(self.df_dataset_historic)} registros")
            
            # Intentar cargar año actual
            try:
                print(f"📥 Descargando dataset año actual...")
                self.df_dataset_current_year = pd.read_csv(current_url)
                print(f"✅ Dataset año actual cargado: {len(self.df_dataset_current_year)} registros")
                self.df_dataset = pd.concat([self.df_dataset_historic, self.df_dataset_current_year])
            except:
                print("⚠️ No se pudo cargar dataset del año actual, usando solo histórico")
                self.df_dataset = self.df_dataset_historic
            
            # Limpieza
            self.df_dataset["season"] = self.df_dataset["season"].astype(str)
            self.df_dataset["Performance_Save%"].fillna(0, inplace=True)
            self.df_dataset['date'] = pd.to_datetime(self.df_dataset['date'])

            self.df_dataset = self.df_dataset.sort_values(by='date',ascending=True)
            
            print(f"✅ Total registros: {len(self.df_dataset)}")
            
        except Exception as e:
            raise FileNotFoundError(
                f"\n❌ ERROR: No se pudieron cargar los datos desde GitHub\n"
                f"   Error: {str(e)}\n\n"
                f"💡 Verifica que los archivos existan en el repositorio\n"
            )

    def init_variables(self):
        self.lst_years = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
        print("✅ Variables inicializadas")

    def consume_model_batch(self,partidos,jornada,temporada,league_code):

        df_predict = predecir_partidos_batch(
            partidos=partidos,
            jornada=jornada,
            temporada=temporada,
            league_code=league_code,
            export_csv=True,
            filename=f"results\{league_code}\{league_code}-{temporada}-{jornada}-predicciones.csv",
            df_database = self.df_dataset,
            xgb_model = self.xgb_model,
            scaler=self.scaler,
            lst_years=self.lst_years
        )

        # Mostrar resumen
        return df_predict
    
    def consume_model_single(self,local,visitante,jornada,temporada,league_code):

         return predecir_corners(
            local=local,
            visitante=visitante,
            jornada=jornada,
            temporada=temporada,
            league_code=league_code,
            df_database = self.df_dataset,
            xgb_model = self.xgb_model,
            xgb_model_local= self.xgb_model_local,
            xgb_model_away = self.xgb_model_away,
            xgb_model_xg= self.xgb_model_xg,
            xgb_model_xg_local= self.xgb_model_xg_local,
            xgb_model_xg_away= self.xgb_model_xg_away,
            xgb_model_gf = self.xgb_model_gf,
            xgb_model_gf_local = self.xgb_model_gf_local,
            xgb_model_gf_away = self.xgb_model_gf_away,
            xgb_model_st = self.xgb_model_st,
            xgb_model_st_local = self.xgb_model_st_local,
            xgb_model_st_away = self.xgb_model_st_away,
            scaler=self.scaler,
            lst_years=self.lst_years
        )


    def kelly_stats(self,p, odds, fraction=0.2):

        b = odds - 1
        q = 1 - p
        f_star = (b * p - q) / b
        f_star = max(f_star, 0)  # evita negativos
        return f_star * fraction  # usa 0.1 para Kelly 10%
