import pandas as pd





def get_ck(df, season, round_num, local, away, league=None):
    """Obtiene corners totales de un partido específico"""
    season_round = (df['season'] == season) & (df['round'] == round_num)
    
    if league is not None:
        season_round = season_round & (df['league'] == league)
    
    df = df[season_round]
    
    df_local = df[df['team'] == local]
    df_away = df[df['team'] == away]
    
    total_ck = df_local["Pass Types_CK"].sum() + df_away["Pass Types_CK"].sum()
    
    return total_ck

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

def get_points_from_result(result):
    """Convierte resultado (W/D/L) a puntos"""
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

# ✅ NUEVA FUNCIÓN: Calcular PPP (Puntos Por Partido)
def get_team_ppp(df, team, season, round_num, league=None):
    """
    Calcula puntos por partido (PPP) de un equipo
    
    Args:
        df: DataFrame completo
        team: Nombre del equipo
        season: Temporada
        round_num: Número de jornada (NO incluye esta jornada)
        league: Código de liga (opcional)
    
    Returns:
        float: Puntos por partido (0-3)
    """
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

# ✅ NUEVA FUNCIÓN: Calcular diferencia de PPP
def get_ppp_difference(df, local, away, season, round_num, league=None):
    """
    Calcula la diferencia de puntos por partido entre local y visitante
    
    Args:
        df: DataFrame completo
        local: Equipo local
        away: Equipo visitante
        season: Temporada
        round_num: Jornada actual
        league: Código de liga (opcional)
    
    Returns:
        float: Diferencia de PPP (local - away)
    """
    local_ppp = get_team_ppp(df, local, season, round_num, league)
    away_ppp = get_team_ppp(df, away, season, round_num, league)
    
    return local_ppp - away_ppp

def get_average(df, is_team=False, lst_avg=None):
    """Calcula promedios de estadísticas"""
    
    if len(df) == 0:
        # Retornar valores por defecto si el DataFrame está vacío
        if is_team:
            return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return (0, 0, 0, 0, 0, 0, 0, 0)
    
    if is_team:
        # ===========================
        # ESTADÍSTICAS BÁSICAS (NORMALIZADAS)
        # ===========================
        avg_cross = (df['Performance_Crs'].sum() / len(df)) - lst_avg[3]
        avg_att_3rd = (df['Touches_Att 3rd'].sum() / len(df)) - lst_avg[4]
        avg_sca = (df['SCA Types_SCA'].sum() / len(df)) - lst_avg[2]
        avg_xg = (df['Expected_xG'].sum() / len(df)) - lst_avg[1]
        
        # ✅ CAMBIO: VARIANZA EN VEZ DE PROMEDIO DE CK
        var_ck = df['Pass Types_CK'].var() if len(df) > 1 else 0
        avg_ck = (df['Pass Types_CK'].sum() / len(df)) - lst_avg[8]
        
        avg_poss = (df['Poss'].sum() / len(df)) - 50
        avg_gf = (df['GF'].sum() / len(df)) - lst_avg[5]
        avg_ga = (df['GA'].sum() / len(df)) - lst_avg[6]
        
        # ===========================
        # MÉTRICAS OFENSIVAS AVANZADAS
        # ===========================
        
        # Precisión de tiros
        total_sh = df['Standard_Sh'].sum()
        sh_accuracy = (df['Standard_SoT'].sum() / total_sh) if total_sh > 0 else 0
        
        # Eficiencia xG por tiro
        xg_shot = (df['Expected_xG'].sum() / total_sh) if total_sh > 0 else 0
        
        # Presencia atacante (% toques en área rival)
        total_touches = df['Touches_Touches'].sum()
        attacking_presence = (df['Touches_Att 3rd'].sum() / total_touches) if total_touches > 0 else 0
        
        # Tiros por posesión
        total_poss = df['Poss'].sum()
        possession_shot = (total_sh / total_poss) if total_poss > 0 else 0
        
        # Distancia promedio de tiros
        standard_dist = df['Standard_Dist'].mean() if 'Standard_Dist' in df.columns else 0
        
        # ===========================
        # MÉTRICAS DE CREACIÓN
        # ===========================
        
        # Ratio de pases progresivos
        total_passes = df['Total_Att'].sum()
        progressive_pass_ratio = (df['PrgP'].sum() / total_passes) if total_passes > 0 else 0
        
        # Participación en último tercio
        final_third_passes = df['1/3'].sum()
        final_third_involvement = (final_third_passes / total_passes) if total_passes > 0 else 0
        
        # Ratio de pases largos
        long_ball_ratio = (df['Long_Att'].sum() / total_passes) if total_passes > 0 else 0
        
        # Asistencias por SCA
        total_sca = df['SCA Types_SCA'].sum()
        assist_sca = (df['Ast'].sum() / total_sca) if total_sca > 0 else 0
        
        # Dependencia de centros
        cross_dependency = (df['Performance_Crs'].sum() / total_passes) if total_passes > 0 else 0
        
        # Eficiencia creativa
        creative_efficiency = (total_sca / total_poss) if total_poss > 0 else 0
        
        # ===========================
        # MÉTRICAS DEFENSIVAS
        # ===========================
        
        # Intensidad de presión alta
        total_tackles = df['Tackles_Tkl'].sum()
        high_press_intensity = (df['Tackles_Att 3rd'].sum() / total_tackles) if total_tackles > 0 else 0
        
        # Ratio intercepciones/tackles
        interception_tackle = (df['Int'].sum() / total_tackles) if total_tackles > 0 else 0
        
        # Ratio bloqueos/tackles
        blocks_tackle = (df['Blocks_Blocks'].sum() / total_tackles) if total_tackles > 0 else 0
        
        # Ratio de despejes
        total_defensive_actions = total_tackles + df['Int'].sum()
        clearance_ratio = (df['Clr'].sum() / total_defensive_actions) if total_defensive_actions > 0 else 0
        
        # ===========================
        # MÉTRICAS DE PORTERÍA
        # ===========================
        
        # Rendimiento del portero normalizado
        avg_save_pct = df['Performance_Save%'].mean() if 'Performance_Save%' in df.columns else 0
        avg_xg_against = df['Expected_xG'].mean() if len(df) > 0 else 1
        performance_save = (avg_save_pct / (1 / avg_xg_against)) if avg_xg_against > 0 else 0
        
        # ===========================
        # MÉTRICAS DE POSESIÓN
        # ===========================
        
        # Ratio de conducciones progresivas
        total_carries = df['Carries_Carries'].sum()
        progressive_carry_ratio = (df['Carries_PrgC'].sum() / total_carries) if total_carries > 0 else 0
        
        # Ratio de conducciones al área
        penalty_carry_ratio = (df['Carries_CPA'].sum() / total_carries) if total_carries > 0 else 0
        
        # Balance conducción/pase progresivo
        total_prog_passes = df['PrgP'].sum()
        carry_pass_balance = (df['Carries_PrgC'].sum() / total_prog_passes) if total_prog_passes > 0 else 0
        
        # ===========================
        # ÍNDICES COMPUESTOS
        # ===========================
        
        # Índice ofensivo
        avg_gf_raw = df['GF'].mean()
        avg_xg_raw = df['Expected_xG'].mean()
        avg_sot = df['Standard_SoT'].mean()
        avg_sh = df['Standard_Sh'].mean()
        offensive_index = (avg_gf_raw + avg_xg_raw) * (avg_sot / avg_sh) if avg_sh > 0 else 0
        
        # Índice defensivo
        avg_int = df['Int'].mean()
        avg_tkl = df['Tackles_Tkl'].mean()
        avg_clr = df['Clr'].mean()
        defensive_index = avg_save_pct * (avg_int / (avg_tkl + avg_clr)) if (avg_tkl + avg_clr) > 0 else 0
        
        # Índice de control de posesión
        avg_touches_att = df['Touches_Att 3rd'].mean()
        avg_carries_third = df['Carries_1/3'].mean() if 'Carries_1/3' in df.columns else 0
        avg_touches_total = df['Touches_Touches'].mean()
        possession_control_index = ((avg_touches_att + avg_carries_third) / avg_touches_total) if avg_touches_total > 0 else 0
        
        # Índice de transición
        avg_prgp = df['PrgP'].mean()
        avg_prgc = df['Carries_PrgC'].mean()
        avg_poss_raw = df['Poss'].mean()
        transition_index = ((avg_prgp + avg_prgc) / avg_poss_raw) if avg_poss_raw > 0 else 0
        
        # ✅ RETORNAR TODAS LAS MÉTRICAS (23 valores)
        return (
            avg_ck,
            var_ck,  # 0 - ✅ CAMBIADO: varianza en vez de promedio
            avg_xg,  # 1
            avg_sca,  # 2
            avg_cross,  # 3
            avg_poss,  # 4
            avg_att_3rd,  # 5
            avg_gf,  # 6
            avg_ga,  # 7
            sh_accuracy,  # 8
            xg_shot,  # 9
            attacking_presence,  # 10
            possession_shot,  # 11
            progressive_pass_ratio,  # 12
            final_third_involvement,  # 13
            assist_sca,  # 14
            creative_efficiency,  # 15
            high_press_intensity,  # 16
            interception_tackle,  # 17
            clearance_ratio,  # 18
            progressive_carry_ratio,  # 19
            carry_pass_balance,  # 20
            offensive_index,  # 21
            transition_index  # 22
        )
    
    # ===========================
    # PROMEDIOS DE LIGA (is_team=False)
    # ===========================
    
    avg_cross = df['Performance_Crs'].mean()
    avg_att_3rd = df['Touches_Att 3rd'].mean()
    avg_sca = df['SCA Types_SCA'].mean()
    avg_xg = df['Expected_xG'].mean()
    
    # ✅ CAMBIO: VARIANZA EN VEZ DE PROMEDIO DE CK
    var_ck = df['Pass Types_CK'].var() if len(df) > 1 else 0
    avg_ck = df['Pass Types_CK'].mean()
    
    avg_gf = df['GF'].mean()
    avg_ga = df['GA'].mean()
    
    # ✅ AGREGAR MÉTRICAS BÁSICAS PARA NORMALIZACIÓN
    avg_sh = df['Standard_Sh'].mean() if 'Standard_Sh' in df.columns else 0
    
    return (
        
        var_ck,  # 0 - ✅ CAMBIADO
        avg_xg,  # 1
        avg_sca,  # 2
        avg_cross,  # 3
        avg_att_3rd,  # 4
        avg_gf,  # 5
        avg_ga,  # 6
        avg_sh,  # 7 - NUEVO
        avg_ck
    )

# ===========================
# PROCESAMIENTO DE DATOS CON PPP Y NUEVAS MÉTRICAS
# ===========================

lst_years = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
lst_data = []
y = []

USE_ONE_HOT_ENCODING = True

print("\n🔄 PROCESANDO DATOS CON PPP Y MÉTRICAS AVANZADAS...")
print("=" * 80)

for i in lst_1:
    
    if i[2] < 5:
        continue
    
    local = i[0]
    away = i[1]
    round_num = i[2]
    season = i[3]
    date = i[4]
    league_code = i[5]
    
    dic_df = {}
    
    # Promedios de liga
    lst_avg = get_average(
        df_database[
            (df_database['season'] == season) & 
            (df_database['round'] < round_num) &
            (df_database['league'] == league_code)
        ],
        is_team=False
    )
    
    # ✅ FUNCIÓN MEJORADA: Maneja métricas originales y avanzadas
    def create_line(df, is_form=True, is_team=False, use_advanced=True):
        """
        Args:
            df: DataFrame con datos del equipo
            is_form: Si True, toma solo últimos 8 partidos
            is_team: Si True, normaliza contra promedios de liga
            use_advanced: Si True, incluye métricas avanzadas (23 valores)
                         Si False, solo métricas originales (8 valores)
        """
        if is_form:
            df = df[-6:]
        
        if use_advanced:
            # Retorna 23 valores (todas las métricas)
            return get_average(df, is_team, lst_avg)
        else:
            # Retorna solo 8 valores originales
            result = get_average(df, is_team, lst_avg)
            return result[:9]  # Primeros 8 valores
    
    # Extraer DataFrames
    (team1_home, team1_away, team1_opp_home, team1_opp_away,
     team2_home, team2_away, team2_opp_home, team2_opp_away) = get_dataframes(
        df_database, season, round_num, local, away, league=league_code
    )
    
    # Corners reales
    ck = get_ck(df_database, season, round_num, local, away, league=league_code)
    y.append(ck)
    
    # Head to Head
    index = lst_years.index(season)
    result = lst_years[:index+1]
    team1_h2h, team2_h2h = get_head_2_head(
        df_database, local, away, seasons=result, league=league_code
    )
    
    # ✅ PPP
    local_ppp = get_team_ppp(df_database, local, season, round_num, league=league_code)
    away_ppp = get_team_ppp(df_database, away, season, round_num, league=league_code)
    ppp_diff = local_ppp - away_ppp
    
    dic_df['ppp_local'] = (local_ppp,)
    dic_df['ppp_away'] = (away_ppp,)
    dic_df['ppp_difference'] = (ppp_diff,)
    
    # ✅ FEATURES CON MÉTRICAS AVANZADAS (23 valores cada una)
    dic_df['lst_team1_home_form'] = create_line(team1_home, True, True, use_advanced=True)
    dic_df['lst_team1_home_general'] = create_line(team1_home, False, True, use_advanced=True)
    dic_df['lst_team1_away_form'] = create_line(team1_away, True, True, use_advanced=True)
    dic_df['lst_team1_away_general'] = create_line(team1_away, False, True, use_advanced=True)
    
    dic_df['lst_team2_home_form'] = create_line(team2_home, True, True, use_advanced=True)
    dic_df['lst_team2_home_general'] = create_line(team2_home, False, True, use_advanced=True)
    dic_df['lst_team2_away_form'] = create_line(team2_away, True, True, use_advanced=True)
    dic_df['lst_team2_away_general'] = create_line(team2_away, False, True, use_advanced=True)
    
    dic_df['lst_team1_h2h'] = create_line(team1_h2h, False, True, use_advanced=True)
    dic_df['lst_team2_h2h'] = create_line(team2_h2h, False, True, use_advanced=True)
    
    # ✅ FEATURES CON MÉTRICAS ORIGINALES (8 valores) - SOLO PARA OPONENTES
    dic_df['lst_team1_opp_away'] = create_line(team1_opp_away, False, True, use_advanced=False)
    dic_df['lst_team2_opp_home'] = create_line(team2_opp_home, False, True, use_advanced=False)
    
    # One-Hot Encoding
    if USE_ONE_HOT_ENCODING:
        league_dummies = {
            'league_ESP': 1 if league_code == 'ESP' else 0,
            'league_GER': 1 if league_code == 'GER' else 0,
            'league_FRA': 1 if league_code == 'FRA' else 0,
            'league_ITA': 1 if league_code == 'ITA' else 0,
            'league_NED': 1 if league_code == 'NED' else 0
        }
        
        for key, value in league_dummies.items():
            dic_df[key] = (value,)
    
    # ✅ CONSTRUIR VECTOR DE FEATURES CON NOMBRES DESCRIPTIVOS
    lst_base_advanced = [
        "avg_ck","var_ck",  # ✅ CAMBIADO
        "xg", "sca", "cross", "poss", "att_3rd", "gf", "ga",
        "sh_accuracy", "xg_shot", "attacking_presence", "possession_shot",
        "progressive_pass_ratio", "final_third_involvement", "assist_sca", "creative_efficiency",
        "high_press_intensity", "interception_tackle", "clearance_ratio",
        "progressive_carry_ratio", "carry_pass_balance", "offensive_index", "transition_index"
    ]
    
    lst_base_original = [
        "var_ck","xg", "sca", "cross", "poss", "att_3rd", "gf", "ga","avg_ck"
    ]
    
    lst_features_values = []
    lst_features_names = []
    
    for key in dic_df:
        lst_features_values.extend(list(dic_df[key]))
        
        # Casos especiales
        if key in ['ppp_local', 'ppp_away', 'ppp_difference']:
            lst_features_names.append(key)
        elif key.startswith('league_'):
            lst_features_names.append(key)
        elif key in ['lst_team1_opp_away', 'lst_team2_opp_home']:
            # ✅ Métricas ORIGINALES (8 valores)
            lst_features_names.extend([f"{key}_{col}" for col in lst_base_original])
        else:
            # ✅ Métricas AVANZADAS (23 valores)
            lst_features_names.extend([f"{key}_{col}" for col in lst_base_advanced])
    
    lst_data.append(lst_features_values)

# ===========================
# CREAR DATAFRAME FINAL
# ===========================

df_data = pd.DataFrame(data=lst_data, columns=lst_features_names)

print(f"\n✅ PROCESAMIENTO COMPLETADO:")
print(f"   Shape inicial: {df_data.shape}")
print(f"   Total partidos: {len(df_data)}")
print(f"   Features totales: {df_data.shape[1]}")

# ===========================
# LIMPIEZA DE DATOS NULOS
# ===========================

print(f"\n🧹 LIMPIANDO DATOS NULOS...")

import numpy as np
nulos_antes_X = df_data.isnull().sum().sum()
nulos_antes_y = np.isnan(y).sum() if isinstance(y, np.ndarray) else sum(pd.isna(y))

print(f"   Nulos en X (antes): {nulos_antes_X}")
print(f"   Nulos en Y (antes): {nulos_antes_y}")

y_array = np.array(y).flatten()

mask_valid_X = ~df_data.isnull().any(axis=1)
mask_valid_y = ~np.isnan(y_array)
mask_combined = mask_valid_X & mask_valid_y

df_data = df_data[mask_combined].reset_index(drop=True)
y_array = y_array[mask_combined]

print(f"\n✅ LIMPIEZA COMPLETADA:")
print(f"   Nulos en X (después): {df_data.isnull().sum().sum()}")
print(f"   Nulos en Y (después): {np.isnan(y_array).sum()}")
print(f"   Filas eliminadas: {len(mask_combined) - mask_combined.sum()}")
print(f"   Shape final: {df_data.shape}")

# ===========================
# VERIFICACIÓN FINAL
# ===========================

print(f"\n🔍 VERIFICACIÓN DE NUEVAS FEATURES:")
print(f"   ✅ Features con 'var_ck': {len([c for c in df_data.columns if 'var_ck' in c])}")
print(f"   ✅ Features con métricas avanzadas: {len([c for c in df_data.columns if any(m in c for m in ['sh_accuracy', 'offensive_index'])])}")
print(f"   ✅ Features de oponentes (8 valores): {len([c for c in df_data.columns if 'opp' in c])}")

y = y_array.tolist()

print("\n" + "=" * 80)
print("✅ PROCESO COMPLETADO - DATOS LISTOS PARA ENTRENAMIENTO")
print("=" * 80)