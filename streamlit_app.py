import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats as scipy_stats
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(layout="wide", page_title="Corners Forecast", page_icon="⚽")

# 👈 AÑADIR MARGEN AL LAYOUT WIDE
st.markdown("""
    <style>
        .block-container {
            padding-left: 5rem;
            padding-right: 5rem;
            max-width: 1400px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# --- CONSTANTES DEL MODELO ---
MSE_MODELO = 1.99
RMSE_MODELO = 2.4
R2_MODELO = 0.39
N_SIMULACIONES = 5000

# --- ERRORES ESTIMADOS POR MODELO (RMSE) ---
# Corners
RMSE_CK_TOTAL = 1.99
RMSE_CK_LOCAL = 1.64
RMSE_CK_AWAY = 1.45

# Goles
RMSE_GF_TOTAL = .95
RMSE_GF_LOCAL = .6
RMSE_GF_AWAY = .6

# xG (Goles Esperados)
RMSE_XG_TOTAL = 1
RMSE_XG_LOCAL = .6
RMSE_XG_AWAY = .6

# Tiros a Puerta (Shots on Target)
RMSE_ST_TOTAL = 1.7
RMSE_ST_LOCAL = 1.4
RMSE_ST_AWAY = 1.3

# --- FUNCIONES AUXILIARES ---
def probabilidad_a_momio(probabilidad):
    """Convierte probabilidad (%) a momio decimal"""
    if probabilidad <= 0:
        return 0
    return round(100 / probabilidad, 2)

def clasificar_valor_apuesta(momio_real, momio_modelo):
    """Determina si hay valor en la apuesta"""
    if momio_real > momio_modelo * 1.1:
        return "🟢 EXCELENTE VALOR"
    elif momio_real > momio_modelo:
        return "🟡 BUEN VALOR"
    else:
        return "🔴 SIN VALOR"

@st.cache_data(ttl=3600)
def simular_lambda_montecarlo(lambda_pred, sigma=RMSE_MODELO, n_sims=N_SIMULACIONES):
    """Genera simulaciones Monte Carlo con CACHE"""
    lambdas = np.random.normal(lambda_pred, sigma, n_sims)
    lambdas = np.maximum(lambdas, 0.1)
    return lambdas

@st.cache_data(ttl=3600)
def calcular_probabilidades_con_incertidumbre(lambda_pred, linea, tipo='over', sigma=RMSE_MODELO, n_sims=N_SIMULACIONES):
    """Calcula probabilidades con CACHE"""
    lambdas_sim = simular_lambda_montecarlo(lambda_pred, sigma, n_sims)
    probs = []
    
    if tipo == 'over':
        for lam in lambdas_sim:
            prob = 1 - scipy_stats.poisson.cdf(int(linea), lam)
            probs.append(prob * 100)
    else:
        for lam in lambdas_sim:
            prob = scipy_stats.poisson.cdf(int(linea) - 1, lam)
            probs.append(prob * 100)
    
    probs = np.array(probs)
    
    return {
        'prob_media': np.mean(probs),
        'prob_low': np.percentile(probs, 5),
        'prob_high': np.percentile(probs, 95),
        'prob_std': np.std(probs),
        'distribucion': probs
    }

def calcular_expected_value(prob_media, momio_casa):
    """Calcula Expected Value (EV)"""
    prob_decimal = prob_media / 100
    ev = (prob_decimal * momio_casa) - 1
    return ev * 100

def calcular_kelly_criterion(prob_media, momio_casa):
    """Calcula Kelly Criterion"""
    p = prob_media / 100
    
    if momio_casa <= 1:
        return 0
    
    kelly = (p * momio_casa - 1) / (momio_casa - 1)
    
    if kelly < 0:
        return 0
    
    return min(kelly, 0.25)

def recomendar_apuesta_avanzada(prob_media, prob_low, prob_high, momio_casa):
    """Sistema avanzado de recomendación"""
    prob_casa = (1 / momio_casa) * 100
    ev = calcular_expected_value(prob_media, momio_casa)
    kelly = calcular_kelly_criterion(prob_media, momio_casa)
    kelly_conservador = kelly * 0.25
    
    ev_positivo = ev > 0
    confianza_alta = prob_low > prob_casa
    margen_seguridad = (prob_media - prob_casa) / prob_casa
    
    if confianza_alta and ev > 5 and margen_seguridad > 0.1:
        nivel = "EXCELENTE"
        emoji = "🟢"
        recomendar = True
    elif confianza_alta and ev > 0:
        nivel = "BUENA"
        emoji = "🟡"
        recomendar = True
    elif ev > 0:
        nivel = "MODERADA"
        emoji = "🟠"
        recomendar = False
    else:
        nivel = "MALA"
        emoji = "🔴"
        recomendar = False
    
    return {
        'recomendar': recomendar,
        'nivel': nivel,
        'emoji': emoji,
        'ev': ev,
        'kelly': kelly * 100,
        'kelly_conservador': kelly_conservador * 100,
        'prob_casa': prob_casa,
        'prob_media': prob_media,
        'prob_low': prob_low,
        'prob_high': prob_high,
        'margen_seguridad': margen_seguridad * 100,
        'ev_positivo': ev_positivo,
        'confianza_alta': confianza_alta
    }

# --- DICCIONARIO DE LIGAS ---
LEAGUES_DICT = {
    "Ligue 1": "FRA",
    "La Liga": "ESP",
    "Premier League": "ENG",
    "Eredivisie": "NED",
    "Liga NOS": "POR",
    "Pro League": "BEL",
    "Bundesliga": "GER",
    "Serie A": "ITA"
}

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>Corners Forecast</h1>", unsafe_allow_html=True)

# --- CARGAR DATOS ---
@st.cache_data
def cargar_datos():
    df_historic = pd.read_csv(r"https://raw.githubusercontent.com/danielsaed/futbol_corners_forecast/refs/heads/main/dataset/cleaned/dataset_cleaned.csv")
    df_current_year = pd.read_csv(r"https://raw.githubusercontent.com/danielsaed/futbol_corners_forecast/refs/heads/main/dataset/cleaned/dataset_cleaned_current_year.csv")

    df = pd.concat([df_historic,df_current_year])
    return df[['local','league','season']].drop_duplicates()

df = cargar_datos()

# --- INICIALIZAR SESSION STATE ---
if 'prediccion_realizada' not in st.session_state:
    st.session_state.prediccion_realizada = False
if 'resultado_api' not in st.session_state:
    st.session_state.resultado_api = None

# 👇 NUEVO: Guardar valores anteriores para detectar cambios
if 'prev_liga' not in st.session_state:
    st.session_state.prev_liga = None
if 'prev_jornada' not in st.session_state:
    st.session_state.prev_jornada = None
if 'prev_temporada' not in st.session_state:
    st.session_state.prev_temporada = None
if 'prev_local' not in st.session_state:
    st.session_state.prev_local = None
if 'prev_away' not in st.session_state:
    st.session_state.prev_away = None

st.markdown("")

# --- SELECCIÓN DE PARÁMETROS ---
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    option = st.selectbox(
        "🏆 Liga",
        ["La Liga", "Premier League", "Ligue 1", "Serie A", "Eredivisie", "Liga NOS", "Pro League", "Bundesliga"],
        index=None,
        placeholder="Selecciona liga",
        key="liga_select"
    )

# 👇 DETECTAR CAMBIO EN LIGA
if option != st.session_state.prev_liga:
    st.session_state.prediccion_realizada = False
    st.session_state.resultado_api = None
    st.session_state.prev_liga = option

st.write("")

col_jornada1, col_jornada2, col_jornada3, col_jornada4 = st.columns([2, 1, 1, 2])

jornada = None
temporada = None

with col_jornada2:
    if option:
        jornada = st.number_input("📅 Jornada", min_value=5, max_value=42, value=15, step=1, key="jornada_input")
        
        # 👇 DETECTAR CAMBIO EN JORNADA
        if jornada != st.session_state.prev_jornada:
            st.session_state.prediccion_realizada = False
            st.session_state.resultado_api = None
            st.session_state.prev_jornada = jornada

with col_jornada3:
    if option:
        temporada = st.selectbox(
            "Temporada",
            [2526, 2425, 2324, 2223, 2122],
            index=0,
            key="temporada_select"
        )
        
        # 👇 DETECTAR CAMBIO EN TEMPORADA
        if temporada != st.session_state.prev_temporada:
            st.session_state.prediccion_realizada = False
            st.session_state.resultado_api = None
            st.session_state.prev_temporada = temporada

st.write("")

cl2, cl3, cl4 = st.columns([4, 1, 4])

option_local = None
option_away = None

with cl2:
    if option:
        if jornada:
            option_local = st.selectbox(
                "🏠 Equipo Local",
                list(df["local"][(df["league"] == LEAGUES_DICT[option]) & (df["season"] == temporada)]),
                index=None,
                placeholder="Equipo local",
                key="local_select"
            )
            
            # 👇 DETECTAR CAMBIO EN EQUIPO LOCAL
            if option_local != st.session_state.prev_local:
                st.session_state.prediccion_realizada = False
                st.session_state.resultado_api = None
                st.session_state.prev_local = option_local

with cl3:
    if option:
        st.write("")
        st.write("")
        st.markdown("<h3 style='text-align: center'>VS</h3>", unsafe_allow_html=True)

with cl4:
    if option:
        if jornada:
            option_away = st.selectbox(
                "✈️ Equipo Visitante",
                list(df["local"][(df["league"] == LEAGUES_DICT[option]) & (df["season"] == temporada)]),
                index=None,
                placeholder="Equipo visitante",
                key="away_select"
            )
            
            # 👇 DETECTAR CAMBIO EN EQUIPO VISITANTE
            if option_away != st.session_state.prev_away:
                st.session_state.prediccion_realizada = False
                st.session_state.resultado_api = None
                st.session_state.prev_away = option_away

# --- BOTÓN PARA GENERAR PREDICCIÓN ---
if option and option_local and option_away:
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        if st.button("Generar Predicción", type="secondary", use_container_width=True):
            st.session_state.prediccion_realizada = True
            st.session_state.resultado_api = None
    
    st.write("")
    st.write("")

# --- REALIZAR PREDICCIÓN (SOLO SI SE PRESIONÓ EL BOTÓN) ---
if option and option_local and option_away and st.session_state.prediccion_realizada:
    
    if st.session_state.resultado_api is None:
        
        with st.spinner('🔮 Generando predicción con análisis de incertidumbre...'):
            
            url = "https://daniel-saed-futbol-corners-forecast-api.hf.space/items/"
            #url = "http://localhost:7860/items/"
            headers = {"X-API-Key": API_KEY}
            params = {
                "local": option_local,
                "visitante": option_away,
                "jornada": jornada,
                "league_code": LEAGUES_DICT[option],
                "temporada": str(temporada)
            }
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                if response.status_code == 200:
                    st.session_state.resultado_api = response.json()
                    st.success("✅ Predicción generada")
                    
                elif response.status_code == 401:
                    st.error("❌ Error de Autenticación - API Key inválida")
                    st.stop()
                elif response.status_code == 400:
                    st.error(f"❌ Error: {response.json().get('detail', 'Parámetros inválidos')}")
                    st.stop()
                else:
                    st.error(f"❌ Error {response.status_code}")
                    st.stop()
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Timeout - Intenta de nuevo")
                st.stop()
            except requests.exceptions.ConnectionError:
                st.error("🌐 Error de conexión")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
    
    # --- MOSTRAR RESULTADOS ---
    if st.session_state.resultado_api:
        resultado = st.session_state.resultado_api
        lambda_pred = resultado['prediccion']
        
        # Extraer predicciones detalladas
        pred_ck_total = resultado.get('prediccion', 0)
        pred_ck_local = resultado.get('prediccion_local', 0)
        pred_ck_away = resultado.get('prediccion_away', 0)
        
        pred_xg_total = resultado.get('prediccion_xg', 0)
        pred_xg_local = resultado.get('prediccion_xg_local', 0)
        pred_xg_away = resultado.get('prediccion_xg_away', 0)
        
        pred_gf_total = resultado.get('prediccion_gf', 0)
        pred_gf_local = resultado.get('prediccion_gf_local', 0)
        pred_gf_away = resultado.get('prediccion_gf_away', 0)
        
        pred_st_total = resultado.get('prediccion_st', 0)
        pred_st_local = resultado.get('prediccion_st_local', 0)
        pred_st_away = resultado.get('prediccion_st_away', 0)
        
        st.write("")
        st.write("")
        
        # ============================================
        # 1. PREDICCIONES MACHINE LEARNING
        # ============================================
        
        st.markdown("# Predicciones")
        st.write("")
        st.caption("Modelos XGBoost entrenados con alrededor de 13,000 partidos utilizando metricas avanzadas de futbol de las principales ligas europeas (2018 a 2025). Datos obtenidos de OPTA.")
        
        def mostrar_bloque_prediccion(titulo, total, local, away, rmse_total, rmse_local, rmse_away, icono):
            st.markdown(f"#### {icono} {titulo}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total", f"{total:.2f}", delta=f"± {rmse_total}", delta_color="off", help=f"RMSE estimado: {rmse_total}")
            with c2:
                st.metric(f"Local ({option_local})", f"{local:.2f}", delta=f"± {rmse_local}", delta_color="off", help=f"RMSE estimado: {rmse_local}")
            with c3:
                st.metric(f"Visitante ({option_away})", f"{away:.2f}", delta=f"± {rmse_away}", delta_color="off", help=f"RMSE estimado: {rmse_away}")
            st.divider()

        # 1. Tiros de Esquina
        mostrar_bloque_prediccion(
            "Tiros de esquina", 
            pred_ck_total, pred_ck_local, pred_ck_away, 
            RMSE_CK_TOTAL, RMSE_CK_LOCAL, RMSE_CK_AWAY, 
            "🚩"
        )

        # 2. Goles
        mostrar_bloque_prediccion(
            "Goles", 
            pred_gf_total, pred_gf_local, pred_gf_away, 
            RMSE_GF_TOTAL, RMSE_GF_LOCAL, RMSE_GF_AWAY, 
            "⚽"
        )

        # 3. xG (Goles Esperados)
        mostrar_bloque_prediccion(
            "xG (Goles Esperados)", 
            pred_xg_total, pred_xg_local, pred_xg_away, 
            RMSE_XG_TOTAL, RMSE_XG_LOCAL, RMSE_XG_AWAY, 
            "📈"
        )

        # 4. Tiros a Puerta
        mostrar_bloque_prediccion(
            "Tiros a puerta", 
            pred_st_total, pred_st_local, pred_st_away, 
            RMSE_ST_TOTAL, RMSE_ST_LOCAL, RMSE_ST_AWAY, 
            "🎯")

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        
        # ============================================
        # 2. ANÁLISIS DE EQUIPOS
        # ============================================
        
        # Extraer datos nuevos
# Extraer datos nuevos
 # Extraer datos nuevos
        stats_ck = resultado.get('stats_ck', {})
        stats_gf = resultado.get('stats_gf', {})
        stats_xg = resultado.get('stats_xg', {})
        stats_st = resultado.get('stats_st', {}) # Nuevo
        
        ppp_local = resultado.get('ppp_local', 0)
        ppp_away = resultado.get('ppp_away', 0)
        
        riesgo = resultado['riesgo']
        
        st.markdown("# Stats")
        
        # Métrica de Forma (PPP)
        col_form1, col_form2, col_form3 = st.columns(3)
        with col_form1:
             st.metric("Forma Local (PPP)", f"{ppp_local:.2f}", help="Puntos por Partido")
        with col_form2:
             diff_ppp = ppp_local - ppp_away
             st.metric("Diferencia de Nivel", f"{diff_ppp:.2f}", delta_color="off", help="Diferencia de PPP (Local - Visitante)")
        with col_form3:
             st.metric("Forma Visitante (PPP)", f"{ppp_away:.2f}", help="Puntos por Partido")

        st.write("")

        # --- FUNCIÓN PARA RENDERIZAR PESTAÑAS ---
        def render_stats_tab(stats_data, type_key, label_metric):
            """Renderiza el contenido de una pestaña de estadísticas con el nuevo layout"""
            
            # --- 1. PREPARAR DATOS ---
            # General
            l_h = stats_data.get(f'local_{type_key}_home', 0)
            l_a = stats_data.get(f'local_{type_key}_away', 0)
            a_h = stats_data.get(f'away_{type_key}_home', 0)
            a_a = stats_data.get(f'away_{type_key}_away', 0)
            
            l_rec_h = stats_data.get(f'local_{type_key}_received_home', 0)
            l_rec_a = stats_data.get(f'local_{type_key}_received_away', 0)
            a_rec_h = stats_data.get(f'away_{type_key}_received_home', 0)
            a_rec_a = stats_data.get(f'away_{type_key}_received_away', 0)
            
            # Forma
            l_h_f = stats_data.get(f'local_{type_key}_home_form', 0)
            l_a_f = stats_data.get(f'local_{type_key}_away_form', 0)
            a_h_f = stats_data.get(f'away_{type_key}_home_form', 0)
            a_a_f = stats_data.get(f'away_{type_key}_away_form', 0)
            
            l_rec_h_f = stats_data.get(f'local_{type_key}_received_home_form', 0)
            l_rec_a_f = stats_data.get(f'local_{type_key}_received_away_form', 0)
            a_rec_h_f = stats_data.get(f'away_{type_key}_received_home_form', 0)
            a_rec_a_f = stats_data.get(f'away_{type_key}_received_away_form', 0)
            
            # Globales (Promedio simple Home+Away)
            l_g = (l_h + l_a) / 2
            a_g = (a_h + a_a) / 2
            l_rec_g = (l_rec_h + l_rec_a) / 2
            a_rec_g = (a_rec_h + a_rec_a) / 2
            
            l_g_f = (l_h_f + l_a_f) / 2
            a_g_f = (a_h_f + a_a_f) / 2
            l_rec_g_f = (l_rec_h_f + l_rec_a_f) / 2
            a_rec_g_f = (a_rec_h_f + a_rec_a_f) / 2

            # --- FUNCIÓN AUXILIAR PARA MOSTRAR TABLA CON TOTALES ---
            def display_styled_df(teams, favors, contras):
                df = pd.DataFrame({
                    'Equipo': teams,
                    'A Favor': favors,
                    'En Contra': contras
                })
                
                # Calcular Totales por Fila (Total del equipo)
                df['Total'] = df['A Favor'] + df['En Contra']
                
                # Calcular Totales por Columna (Suma de ambos equipos)
                # NOTA: El total de totales (esquina inferior derecha) se deja vacío
                total_row = pd.DataFrame({
                    'Equipo': ['TOTAL'],
                    'A Favor': [df['A Favor'].sum()],
                    'En Contra': [df['En Contra'].sum()],
                    'Total': [0]
                })
                
                df_final = pd.concat([df, total_row], ignore_index=True)
                
                # Estilos
                # na_rep="" hace que el None se muestre como celda vacía
                styler = df_final.style.format(subset=['A Favor', 'En Contra', 'Total'], formatter="{:.2f}", na_rep="")
                
                # Estilo: Fondo transparente y texto gris
                style_css = 'color: #888888; font-weight: bold;'
                
                # Resaltar última fila (Totales de columna)
                styler.apply(lambda x: [style_css if x.name == df_final.index[-1] else '' for _ in x], axis=1)
                
                # Resaltar columna Total (Totales de fila)
                styler.apply(lambda x: [style_css if x.name == 'Total' else '' for _ in x], axis=0)
                
                st.dataframe(styler, hide_index=True, use_container_width=True)

            # --- 2. RENDERIZAR SECCIÓN GENERAL ---
            st.markdown("#### 📊 Datos Generales (Temporada)")
            c1, c2, c3 = st.columns(3)
            
            # Columna 1: Contexto Real
            with c1:
                st.caption("🏟️ Contexto (Local en Casa / Vis. Fuera)")
                display_styled_df(
                    [f'🏠 {option_local}', f'✈️ {option_away}'],
                    [l_h, a_a],
                    [l_rec_h, a_rec_a]
                )

            # Columna 2: Inversa
            with c2:
                st.caption("🔄 Inversa (Local Fuera / Vis. Casa)")
                display_styled_df(
                    [f'✈️ {option_local}', f'🏠 {option_away}'],
                    [l_a, a_h],
                    [l_rec_a, a_rec_h]
                )

            # Columna 3: Global
            with c3:
                st.caption("🌍 Global (Promedio Total)")
                display_styled_df(
                    [f'{option_local}', f'{option_away}'],
                    [l_g, a_g],
                    [l_rec_g, a_rec_g]
                )

            # --- 3. RENDERIZAR SECCIÓN FORMA ---
            st.markdown("#### 🔥 Estado de Forma (Últimos 6 Partidos)")
            c1_f, c2_f, c3_f = st.columns(3)
            
            # Columna 1: Contexto Forma
            with c1_f:
                st.caption("🏟️ Contexto (Forma)")
                display_styled_df(
                    [f'🏠 {option_local}', f'✈️ {option_away}'],
                    [l_h_f, a_a_f],
                    [l_rec_h_f, a_rec_a_f]
                )

            # Columna 2: Inversa Forma
            with c2_f:
                st.caption("🔄 Inversa (Forma)")
                display_styled_df(
                    [f'✈️ {option_local}', f'🏠 {option_away}'],
                    [l_a_f, a_h_f],
                    [l_rec_a_f, a_rec_h_f]
                )

            # Columna 3: Global Forma
            with c3_f:
                st.caption("🌍 Global (Forma)")
                display_styled_df(
                    [f'{option_local}', f'{option_away}'],
                    [l_g_f, a_g_f],
                    [l_rec_g_f, a_rec_g_f]
                )
            
            # --- 4. RENDERIZAR H2H ---
            st.markdown("#### ⚔️ Head to Head (H2H)")
            h2h_val = stats_data.get(f'h2h_{type_key}_total', 0)
            st.metric(f"Promedio {label_metric} H2H", f"{h2h_val:.2f}")
            

        # Tabs para las diferentes estadísticas
        tab_ck, tab_gf, tab_xg, tab_st = st.tabs(["🚩 Corners", "⚽ Goles", "📈 xG (Esperados)", "🎯 Tiros a Puerta"])
        
        with tab_ck:
            render_stats_tab(stats_ck, 'ck', 'Corners')
        
        with tab_gf:
            render_stats_tab(stats_gf, 'gf', 'Goles')
            
        with tab_xg:
            render_stats_tab(stats_xg, 'xg', 'xG')
            
        with tab_st:
            render_stats_tab(stats_st, 'st', 'Tiros a Puerta')
            
        # --- MOSTRAR TABLA H2H DETALLADA ---
        if 'h2h_matches' in resultado and resultado['h2h_matches']:
            st.markdown("### 📜 Historial de Partidos (H2H)")
            
            h2h_data = []
            for match in resultado['h2h_matches']:
                # Datos del equipo local en ese partido
                home_team = match['match_home_team']
                away_team = match['match_away_team']
                
                # Identificar stats correctas
                if match['local_team_stats']['team'] == home_team:
                    home_stats = match['local_team_stats']
                    away_stats = match['away_team_stats']
                else:
                    home_stats = match['away_team_stats']
                    away_stats = match['local_team_stats']
                
                h2h_data.append({
                    'Temporada': match['season'],
                    'Jornada': match['round'],
                    'Local': home_team,
                    'Visitante': away_team,
                    'Goles L': home_stats['goals'],
                    'Goles V': away_stats['goals'],
                    'Corners L': home_stats['corners'],
                    'Corners V': away_stats['corners'],
                    'xG L': home_stats['xg'],
                    'xG V': away_stats['xg'],
                    'SoT L': home_stats['sot'],
                    'SoT V': away_stats['sot']
                })
            
            df_h2h = pd.DataFrame(h2h_data)
            
            st.dataframe(
                df_h2h, 
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Temporada': st.column_config.TextColumn('📅 Temp', width='small'),
                    'Jornada': st.column_config.NumberColumn('#', width='small', format="%d"),
                    'Goles L': st.column_config.NumberColumn('⚽ L', format="%.0f"),
                    'Goles V': st.column_config.NumberColumn('⚽ V', format="%.0f"),
                    'Corners L': st.column_config.NumberColumn('🚩 L', format="%.0f"),
                    'Corners V': st.column_config.NumberColumn('🚩 V', format="%.0f"),
                    'xG L': st.column_config.NumberColumn('📈 xG L', format="%.2f"),
                    'xG V': st.column_config.NumberColumn('📈 xG V', format="%.2f"),
                    'SoT L': st.column_config.NumberColumn('🎯 SoT L', format="%.0f"),
                    'SoT V': st.column_config.NumberColumn('🎯 SoT V', format="%.0f"),
                }
            )
        st.divider()
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")



        
        st.markdown("# Momios y Valor de Apuesta")
        st.write("")
        st.write("")
        st.markdown("### Fiabilidad")
        
        col_fiab1, col_fiab2, col_fiab3 = st.columns(3)
        
        with col_fiab1:
            st.markdown(f"**🏠 {option_local}**")
            st.write(f"**Score:** {riesgo['score_local']:.0f}/100")
            st.write(f"**Nivel:** {riesgo['nivel_local']}")
            st.write(f"**CV:** {riesgo['cv_local']:.1f}%")
            st.progress(riesgo['score_local'] / 100)
        
        with col_fiab2:
            st.markdown("**📊 Fiabilidad Global**")
            score_promedio = riesgo['score_promedio']
            st.write(f"**Score:** {score_promedio:.0f}/100")
            st.write("")
            
            if score_promedio >= 65:
                st.success("🟢 Fiabilidad MUY ALTA")
            elif score_promedio >= 50:
                st.info("🟡 Fiabilidad ALTA")
            elif score_promedio >= 35:
                st.warning("🟠 Fiabilidad MEDIA")
            else:
                st.error("🔴 Fiabilidad BAJA")
        
        with col_fiab3:
            st.markdown(f"**✈️ {option_away}**")
            st.write(f"**Score:** {riesgo['score_away']:.0f}/100")
            st.write(f"**Nivel:** {riesgo['nivel_away']}")
            st.write(f"**CV:** {riesgo['cv_away']:.1f}%")
            st.progress(riesgo['score_away'] / 100)
        
        st.write("")
        st.write("")
        st.markdown("---")
        st.write("")
        st.write("")
        
        # ============================================
        # 3. PROBABILIDADES
        # ============================================
        
        st.info(f"🔬 **Análisis con {N_SIMULACIONES:,} simulaciones Monte Carlo** considerando RMSE={RMSE_MODELO}")
        
        tab_over, tab_under = st.tabs(["⬆️ OVER", "⬇️ UNDER"])
        
        with tab_over:
            probs_over = resultado['probabilidades_over']
            
            st.markdown("### 📈 Probabilidades Over (con Intervalos de Confianza 90%)")
            
            df_over_incertidumbre = []
            
            with st.spinner('Calculando incertidumbres Over...'):
                for linea_str in sorted(probs_over.keys(), key=float, reverse=True):
                    linea = float(linea_str)
                    
                    resultado_inc = calcular_probabilidades_con_incertidumbre(
                        lambda_pred, linea, tipo='over'
                    )
                    
                    prob_media = resultado_inc['prob_media']
                    prob_low = resultado_inc['prob_low']
                    prob_high = resultado_inc['prob_high']
                    
                    momio_medio = probabilidad_a_momio(prob_media)
                    momio_low = probabilidad_a_momio(prob_high)
                    momio_high = probabilidad_a_momio(prob_low)
                    
                    df_over_incertidumbre.append({
                        'Línea': f"Over {linea_str}",
                        'Prob. Media': f"{prob_media:.1f}%",
                        'IC 90%': f"[{prob_low:.1f}%, {prob_high:.1f}%]",
                        'Momio Justo': f"@{momio_medio:.2f}",
                        'Rango Momio': f"[@{momio_low:.2f} - @{momio_high:.2f}]",
                        'linea_num': linea,
                        'prob_media_raw': prob_media,
                        'prob_low_raw': prob_low,
                        'prob_high_raw': prob_high,
                        'tipo': 'Over'
                    })
            
            df_over_display = pd.DataFrame(df_over_incertidumbre)
            
            st.dataframe(
                df_over_display[['Línea', 'Prob. Media', 'Momio Justo']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Línea': st.column_config.TextColumn('🎯 Línea', width='small'),
                    'Prob. Media': st.column_config.TextColumn('📊 Probabilidad', width='small'),
                    'Momio Justo': st.column_config.TextColumn('💰 Momio', width='small'),
                }
            )
            
            st.write("")
            
            fig_over = go.Figure()
            
            lineas_sorted = sorted([x['linea_num'] for x in df_over_incertidumbre])
            probs_medias = [x['prob_media_raw'] for x in sorted(df_over_incertidumbre, key=lambda x: x['linea_num'])]
            probs_low = [x['prob_low_raw'] for x in sorted(df_over_incertidumbre, key=lambda x: x['linea_num'])]
            probs_high = [x['prob_high_raw'] for x in sorted(df_over_incertidumbre, key=lambda x: x['linea_num'])]
            
            fig_over.add_trace(go.Scatter(
                x=[f"Over {l}" for l in lineas_sorted] + [f"Over {l}" for l in lineas_sorted[::-1]],
                y=probs_high + probs_low[::-1],
                fill='toself',
                fillcolor='rgba(46, 204, 113, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='IC 90%',
                hoverinfo='skip'
            ))
            
            fig_over.add_trace(go.Scatter(
                x=[f"Over {l}" for l in lineas_sorted],
                y=probs_medias,
                mode='lines+markers',
                name='Probabilidad Media',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=10)
            ))
            
            fig_over.update_layout(
                title="Probabilidades Over con Banda de Incertidumbre (Monte Carlo)",
                xaxis_title="Línea",
                yaxis_title="Probabilidad (%)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_over, use_container_width=True)
        
        with tab_under:
            probs_under = resultado['probabilidades_under']
            
            st.markdown("### 📉 Probabilidades Under (con Intervalos de Confianza 90%)")
            
            df_under_incertidumbre = []
            
            with st.spinner('Calculando incertidumbres Under...'):
                for linea_str in sorted(probs_under.keys(), key=float, reverse=True):
                    linea = float(linea_str)
                    
                    resultado_inc = calcular_probabilidades_con_incertidumbre(
                        lambda_pred, linea, tipo='under'
                    )
                    
                    prob_media = resultado_inc['prob_media']
                    prob_low = resultado_inc['prob_low']
                    prob_high = resultado_inc['prob_high']
                    
                    momio_medio = probabilidad_a_momio(prob_media)
                    momio_low = probabilidad_a_momio(prob_high)
                    momio_high = probabilidad_a_momio(prob_low)
                    
                    df_under_incertidumbre.append({
                        'Línea': f"Under {linea_str}",
                        'Prob. Media': f"{prob_media:.1f}%",
                        'IC 90%': f"[{prob_low:.1f}%, {prob_high:.1f}%]",
                        'Momio Justo': f"@{momio_medio:.2f}",
                        'Rango Momio': f"[@{momio_low:.2f} - @{momio_high:.2f}]",
                        'linea_num': linea,
                        'prob_media_raw': prob_media,
                        'prob_low_raw': prob_low,
                        'prob_high_raw': prob_high,
                        'tipo': 'Under'
                    })
            
            df_under_display = pd.DataFrame(df_under_incertidumbre)
            
            st.dataframe(
                df_under_display[['Línea', 'Prob. Media', 'IC 90%', 'Momio Justo', 'Rango Momio']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Línea': st.column_config.TextColumn('🎯 Línea', width='small'),
                    'Prob. Media': st.column_config.TextColumn('📊 Probabilidad', width='small'),
                    'IC 90%': st.column_config.TextColumn('📉 Intervalo 90%', width='medium'),
                    'Momio Justo': st.column_config.TextColumn('💰 Momio', width='small'),
                    'Rango Momio': st.column_config.TextColumn('📈 Rango Momios', width='medium')
                }
            )
            
            st.write("")
            
            fig_under = go.Figure()
            
            lineas_sorted_under = sorted([x['linea_num'] for x in df_under_incertidumbre])
            probs_medias_under = [x['prob_media_raw'] for x in sorted(df_under_incertidumbre, key=lambda x: x['linea_num'])]
            probs_low_under = [x['prob_low_raw'] for x in sorted(df_under_incertidumbre, key=lambda x: x['linea_num'])]
            probs_high_under = [x['prob_high_raw'] for x in sorted(df_under_incertidumbre, key=lambda x: x['linea_num'])]
            
            fig_under.add_trace(go.Scatter(
                x=[f"Under {l}" for l in lineas_sorted_under] + [f"Under {l}" for l in lineas_sorted_under[::-1]],
                y=probs_high_under + probs_low_under[::-1],
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='IC 90%',
                hoverinfo='skip'
            ))
            
            fig_under.add_trace(go.Scatter(
                x=[f"Under {l}" for l in lineas_sorted_under],
                y=probs_medias_under,
                mode='lines+markers',
                name='Probabilidad Media',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=10)
            ))
            
            fig_under.update_layout(
                title="Probabilidades Under con Banda de Incertidumbre (Monte Carlo)",
                xaxis_title="Línea",
                yaxis_title="Probabilidad (%)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_under, use_container_width=True)
        

        st.markdown("---")
        st.write("")

            
        # ============================================
        # 4. CALCULADORA
        # ============================================
        st.markdown("### 💰 Calculadora de Valor")
        
        st.write("")
        
        todas_lineas_datos = {}
        
        for item in df_over_incertidumbre:
            todas_lineas_datos[item['Línea']] = item
        
        for item in df_under_incertidumbre:
            todas_lineas_datos[item['Línea']] = item
        
        todas_lineas_ordenadas = sorted(
            todas_lineas_datos.keys(),
            key=lambda x: (0 if 'Over' in x else 1, float(x.split()[1])),
            reverse=True
        )
        
        col_calc1, col_calc2 = st.columns(2)
        
        with col_calc1:
            linea_calc = st.selectbox(
                "🎯 Selecciona línea",
                todas_lineas_ordenadas,
                key="calc_linea"
            )
        
        with col_calc2:
            momio_casa = st.number_input(
                "💰 Momio del casino",
                min_value=1.01,
                max_value=20.0,
                value=2.0,
                step=0.01,
                key="calc_momio",
                help="Ingresa el momio decimal que ofrece la casa de apuestas"
            )
        
        st.write("")
        
        datos_linea = todas_lineas_datos[linea_calc]
        
        prob_media = datos_linea['prob_media_raw']
        prob_low = datos_linea['prob_low_raw']
        prob_high = datos_linea['prob_high_raw']
        
        recomendacion = recomendar_apuesta_avanzada(
            prob_media, prob_low, prob_high, momio_casa
        )
        
        st.markdown("### 📊 Métricas de la Apuesta")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "Prob. Media",
                f"{prob_media:.1f}%",
                help="Probabilidad media según Monte Carlo"
            )
        
        with col_m2:
            momio_justo = probabilidad_a_momio(prob_media)
            st.metric(
                "Momio Justo",
                f"@{momio_justo:.2f}",
                help="Momio que refleja la probabilidad real"
            )
        
        with col_m3:
            delta_ev = "📈 Positivo" if recomendacion['ev'] > 0 else "📉 Negativo"
            st.metric(
                "Expected Value",
                f"{recomendacion['ev']:+.2f}%",
                delta=delta_ev,
                help="Ganancia esperada por cada $1 apostado"
            )
        
        with col_m4:
            st.metric(
                "Prob. Casino",
                f"{recomendacion['prob_casa']:.1f}%",
                help="Probabilidad implícita del momio del casino"
            )
        
        st.write("")
        st.write("")
        
        st.markdown("### 💵 Gestión de Bankroll (Kelly Criterion)")
        
        col_kelly1, col_kelly2 = st.columns(2)
        
        with col_kelly1:
            if recomendacion['kelly'] > 0:
                st.write(f"**Kelly Completo:** {recomendacion['kelly']:.2f}% del bankroll")
                st.write(f"**Kelly Conservador (1/4):** {recomendacion['kelly_conservador']:.2f}% del bankroll ⭐")
                
                st.write("")
                st.markdown("**Ejemplo con Bankroll de $1,000:**")
                apuesta_kelly = (recomendacion['kelly'] / 100) * 1000
                apuesta_conservador = (recomendacion['kelly_conservador'] / 100) * 1000
                
                st.write(f"- Kelly Completo: **${apuesta_kelly:.2f}**")
                st.write(f"- Conservador: **${apuesta_conservador:.2f}**")
                
                ganancia_potencial = apuesta_conservador * (momio_casa - 1)
                st.write(f"- Ganancia potencial: **${ganancia_potencial:.2f}**")
            else:
                st.error("❌ Kelly = 0 - No apostar")
        
        with col_kelly2:
            st.write(f"**EV:** {recomendacion['ev']:+.2f}%")
            st.write(f"**Margen de Seguridad:** {recomendacion['margen_seguridad']:+.1f}%")
            st.write(f"**IC 90%:** [{prob_low:.1f}%, {prob_high:.1f}%]")
            
            st.write("")
            
            if recomendacion['confianza_alta']:
                st.success("✅ Alta confianza: IC inferior supera prob. casino")
            else:
                st.warning("⚠️ Baja confianza: IC inferior NO supera prob. casino")
            
            if recomendacion['ev'] > 10:
                st.success("🟢 EV excelente (>10%)")
            elif recomendacion['ev'] > 5:
                st.info("🟡 EV bueno (5-10%)")
            elif recomendacion['ev'] > 0:
                st.warning("🟠 EV positivo pero bajo (<5%)")
            else:
                st.error("🔴 EV negativo")
    
        st.write("")
        st.write("")
        st.markdown("---")
        st.caption(f"🤖 XGBoost v4.2 + Monte Carlo | 🎲 {N_SIMULACIONES:,} simulaciones | 📊 RMSE: {RMSE_MODELO} | ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    if option:
        if option_local and option_away:
            pass
        else:
            st.info("👆 Selecciona ambos equipos")
    else:
        st.info("👆 Selecciona una liga para comenzar")

# Sidebar
with st.sidebar:
    st.markdown("# Corners Forecast")
    
    st.markdown("---")
    
    st.markdown("### 🔗 Enlaces")
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/danielsaed/futbol_corners_forecast)
    
    [![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-API-FFD21E?style=flat)](https://huggingface.co/spaces/daniel-saed/futbol-corners-forecast-api)
    """)
    
    st.markdown("---")
    
    st.markdown("### Ligas")
    for league in LEAGUES_DICT.keys():
        st.write(f"• {league}")
    
    if st.button("🗑️ Limpiar Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.prediccion_realizada = False
        st.session_state.resultado_api = None
        st.success("✅ Cache limpiado")
        st.rerun()
    
    st.markdown("---")