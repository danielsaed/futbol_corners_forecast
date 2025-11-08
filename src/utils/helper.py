import pandas as pd
import warnings
import os


def desactivar_advertencias():
    warnings.filterwarnings('ignore')

    # Ignorar warnings específicos de bibliotecas comunes
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    os.environ['PYTHONWARNINGS'] = 'ignore'

    pd.options.mode.chained_assignment = None  # Desactivar SettingWithCopyWarning

    print("Advertencias desactivadas...")