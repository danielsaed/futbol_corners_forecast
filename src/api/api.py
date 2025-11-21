# ===========================
# SISTEMA DE PREDICCIÓN DE CORNERS - OPTIMIZADO PARA APUESTAS (VERSIÓN COMPLETA)
# ===========================

import numpy as np
import pandas as pd
import os
from fastapi.responses import JSONResponse
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from src.api.load import USE_MODEL
#from load import USE_MODEL

load_dotenv() 

model = USE_MODEL()

app = FastAPI()

# ===========================
# CONFIGURACIÓN API KEY
# ===========================

API_KEY = os.getenv("API_KEY")  # ⚠️ CÁMBIALA POR UNA SEGURA
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Validar API Key"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida o faltante"
        )
    return api_key

# ===========================
# HELPER: CONVERTIR NUMPY/PANDAS A TIPOS NATIVOS
# ===========================
def convert_to_native(val):
    """Convierte tipos NumPy/Pandas a tipos nativos de Python"""
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return [convert_to_native(item) for item in val.tolist()]
    elif isinstance(val, dict):
        return {key: convert_to_native(value) for key, value in val.items()}
    elif isinstance(val, (list, tuple)):
        return [convert_to_native(item) for item in val]
    elif isinstance(val, pd.Series):
        return convert_to_native(val.to_dict())
    elif isinstance(val, pd.DataFrame):
        return convert_to_native(val.to_dict(orient='records'))
    elif pd.isna(val):
        return None
    else:
        return val
    



# ===========================
# ENDPOINTS
# ===========================

@app.get("/")
def read_root():
    """Endpoint raíz con información de la API"""
    return {
        "api": "Corners Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/": "Información de la API",
            "/items/": "Predicción de corners (requiere API Key)",
            "/health": "Estado de salud"
        },
        "auth": "Requiere header: X-API-Key"
    }



@app.get("/items/")
def predict_corners(
    local: str,
    visitante: str,
    jornada: int,
    league_code: str,
    temporada: str = "2526",
    api_key: str = Depends(get_api_key)  # ✅ PROTEGIDO
):
    """
    Predecir corners para un partido de fútbol
    
    Args:
        local: Nombre del equipo local (requerido)
        visitante: Nombre del equipo visitante (requerido)
        jornada: Número de jornada (requerido, min: 1)
        league_code: Código de liga (requerido: ESP, GER, FRA, ITA, ENG, NED, POR, BEL)
        temporada: Temporada en formato AABB (default: "2526")
        
    Returns:
        JSON con predicción y análisis completo
        
    Example:
        GET /items/?local=Barcelona&visitante=Real%20Madrid&jornada=15&league_code=ESP&temporada=2526
        Headers: X-API-Key: tu-clave-secreta-aqui
    """
    
    # ===========================
    # VALIDACIONES
    # ===========================
    
    # Validar campos obligatorios
    if not local or not visitante:
        raise HTTPException(
            status_code=400,
            detail="Los parámetros 'local' y 'visitante' son obligatorios"
        )
    
    # Validar jornada
    if jornada < 1:
        raise HTTPException(
            status_code=400,
            detail="La jornada debe ser mayor o igual a 1"
        )
    
    # Validar liga
    valid_leagues = ["ESP", "GER", "FRA", "ITA", "ENG", "NED", "POR", "BEL"]
    if league_code not in valid_leagues:
        raise HTTPException(
            status_code=400,
            detail=f"Liga inválida. Ligas válidas: {', '.join(valid_leagues)}"
        )
    
    # ===========================
    # PREDICCIÓN
    # ===========================
    
    try:
        resultado = model.consume_model_single(
            local=local,
            visitante=visitante,
            jornada=jornada,
            temporada=temporada,
            league_code=league_code
        )
        
        # Verificar si hubo error en la predicción
        if resultado.get("error"):
            raise HTTPException(
                status_code=422,
                detail=f"Error en predicción: {resultado['error']}"
            )
        
        # ✅ CONVERTIR TIPOS NUMPY A NATIVOS
        resultado_limpio = convert_to_native(resultado)
        
        # Agregar metadata
        resultado_limpio["metadata"] = {
            "api_version": "1.0.0",
            "model_version": "v4",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(
            status_code=200,
            content=resultado_limpio
        )
        
    except HTTPException:
        # Re-lanzar excepciones HTTP
        raise
    
    except Exception as e:
        # Capturar cualquier otro error
        import traceback
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc() if app.debug else None
        }
        
        return JSONResponse(
            status_code=500,
            content=error_detail
        )