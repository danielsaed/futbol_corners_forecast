# Mostrar todas las columnas (sin truncar)
import pandas as pd

pd.options.display.max_columns = None

# Mostrar todas las filas (sin truncar)
pd.options.display.max_rows = None

# Establecer el ancho de la columna para evitar truncamiento
pd.options.display.width = 1000

# Formatear números de punto flotante para mostrar hasta 2 decimales
pd.options.display.float_format = '{:.2f}'.format

# Si quieres que se muestre el índice en la salida
pd.options.display.show_dimensions = True

import warnings
warnings.filterwarnings('ignore')

# Ignorar warnings específicos de bibliotecas comunes
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Para XGBoost específicamente
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Para pandas
import pandas as pd
pd.options.mode.chained_assignment = None  # Desactivar SettingWithCopyWarning