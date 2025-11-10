import sys
import os

# Añadir la ruta raíz del proyecto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.utils.helper import desactivar_advertencias

import soccerdata as sd
import pandas as pd


def extract_local(game_str):
    try:
        parts = game_str.split(" ", 1)[1].split("-")
        return parts[0].strip() if len(parts) > 0 else None
    except (IndexError, AttributeError):
        return None

def extract_away(game_str):
    try:
        parts = game_str.split(" ", 1)[1].split("-")
        return parts[1].strip() if len(parts) > 1 else None
    except (IndexError, AttributeError):
        return None


class GENERATE_DATASET():
    def __init__(self,current_year):
        print("Clase GENERATE_DATASET Inicializada")

        desactivar_advertencias()
        self.init_variables()
        self.mergue_raw_data_all_leagues(current_year)
        self.process_and_output_dataset(current_year)


    def init_variables(self):

        #Years to get from datasource
        self.LST_YEARS_CONFIG = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

        self.dic_historic_all_leagues = {
            "ENG": {},
            "ESP": {},
            "GER": {},
            "FRA": {},
            "ITA": {},
            "NED": {},
            "ENG2": {},
            "MEX": {},
            "POR": {},
            "BEL": {}
        }


        self.df_database = pd.DataFrame()

        # Diccionary to name leagues to get from datasource
        self.DIC_LEAGUES_CONFIG = {
            "ENG": {
                "name": "ENG-Premier League",
                "code": "ENG"
            },
            "ENG2": {
                "name": "ENG-Championship",
                "code": "ENG2"
            },
            "MEX": {
                "name": "MEX-Liga MX",
                "code": "MEX"
            },
            "POR": {
                "name": "POR-Primeira Liga",
                "code": "POR"
            },
            "BEL": {
                "name": "BEL-Belgian Pro League",
                "code": "BEL"
            },
            "ESP": {
                "name": "ESP-La Liga",
                "code": "ESP"
            },
            "GER": {
                "name": "GER-Bundesliga",
                "code": "GER"
            },
            "FRA": {
                "name": "FRA-Ligue 1",
                "code": "FRA"
            },
            "ITA": {
                "name": "ITA-Serie A",
                "code": "ITA"
            },
            "NED": {
                "name": "NED-Eredivisie",
                "code": "NED"
            }
        }


        lst_base = ['season','date','game','round','day','venue','team','GF','GA','opponent',"result"]
        lst_columns_shooting = ['Expected_xG','Standard_Sh','Standard_SoT','Standard_Dist']
        lst_columns_passing_type = ['Pass Types_CK']
        lst_columns_passing = ['Total_Att','Long_Att','Ast','1/3','PrgP']
        lst_columns_defensive = ['Tackles_Att 3rd','Tackles_Tkl','Blocks_Blocks','Int','Clr']
        lst_columns_keeper = ['Performance_Save%']
        lst_columns_shot_creation = ['SCA Types_SCA']
        lst_columns_misc = ['Performance_Crs']
        lst_columns_possesion = ['Poss', 'Touches_Att 3rd','Carries_PrgC','Touches_Touches','Touches_Att Pen','Carries_Carries','Carries_1/3','Carries_CPA']

        self.lst_columns_combined = lst_base + lst_columns_passing_type +lst_columns_passing+lst_columns_defensive+lst_columns_shooting+lst_columns_keeper+lst_columns_shot_creation+lst_columns_misc+lst_columns_possesion
        print("-Variables inicializadas")

    def get_raw_data_from_source(self,league,year):

        print(f"\nLiga {league}... 📅 Año {year}...", end=" ")
                    # Extraer equipos local/visitante
        
        if league["name"] in ["NED-Eredivisie","POR-Primeira Liga","ENG-Championship"] and year == 2017:
            return
        
        # Crear scraper para la liga específica
        fbref = sd.FBref(leagues=league["name"], seasons=year)
        
        # Leer estadísticas
        team_season_shooting = fbref.read_team_match_stats(stat_type="shooting",opponent_stats = False)
        team_season_passing_types = fbref.read_team_match_stats(stat_type="passing_types",opponent_stats = False)
        team_season_passing = fbref.read_team_match_stats(stat_type="passing",opponent_stats = False)
        team_season_defensive = fbref.read_team_match_stats(stat_type="defense",opponent_stats = False)
        team_season_goalkeeping = fbref.read_team_match_stats(stat_type="keeper",opponent_stats = False)
        team_season_goal_shot_creation = fbref.read_team_match_stats(stat_type="goal_shot_creation",opponent_stats = False)
        team_season_goal_misc = fbref.read_team_match_stats(stat_type="misc",opponent_stats = False)
        team_season_goal_possession = fbref.read_team_match_stats(stat_type="possession",opponent_stats = False)

        df_concat = pd.concat([team_season_shooting,team_season_passing_types,team_season_passing,team_season_defensive,
                        team_season_goalkeeping,team_season_goal_shot_creation,team_season_goal_misc,team_season_goal_possession], axis=1)
        
        # Reset index
        df_reset = df_concat.copy().reset_index()
        
        # Aplanar MultiIndex
        df_reset.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in df_reset.columns.values
        ]

        # Eliminar duplicados
        df_reset = df_reset.loc[:, ~df_reset.columns.duplicated()]

        df_filtered = df_reset[self.lst_columns_combined]
        
        df_filtered["local"] = df_filtered["game"].apply(extract_local)
        df_filtered["away"] = df_filtered["game"].apply(extract_away)
        
        # Agregar código de liga
        df_filtered["league"] = league["code"]

        df_filtered = df_filtered.loc[:, ~df_filtered.columns.duplicated(keep='first')]
        
        # Verificar valores problemáticos
        problematic = df_filtered[df_filtered["away"].isna()]
        if len(problematic) > 0:
            print(f"⚠️ {len(problematic)} registros con formato incorrecto")
        else:
            print(f"✅ {len(df_filtered)} partidos extraídos")

        return df_filtered

    def mergue_raw_data_all_leagues(self, current_year):

        all_dataframes = []

        
        if current_year == True:
        #Process only current year
            for league_key, league_info in self.DIC_LEAGUES_CONFIG.items():

                self.dic_historic_all_leagues[league_key][self.LST_YEARS_CONFIG[-1]] = self.get_raw_data_from_source(league_info,self.LST_YEARS_CONFIG[-1])
        else:

        #Process all years needed execpt for current year
            for league_key, league_info in self.DIC_LEAGUES_CONFIG.items():
                for year in self.LST_YEARS_CONFIG:
                    if year == 2025:
                        continue
                    self.dic_historic_all_leagues[league_key][year] = self.get_raw_data_from_source(league_info,year)
        
        for league_key, dic_historic in self.dic_historic_all_leagues.items():
            for year, df in dic_historic.items():
                all_dataframes.append(df)
        
        self.df_database = pd.concat(all_dataframes, ignore_index=True)

        print("Dataset conbinado")

    def process_and_output_dataset(self,current_year):
        
        # Filtrar solo Matchweek
        self.df_database = self.df_database[self.df_database['round'].str.contains("Matchweek", na=False)]
        self.df_database['round'] = self.df_database['round'].str.replace("Matchweek ", "")

        # Convertir tipos
        self.df_database['round'] = self.df_database['round'].astype(int)
        self.df_database['GF'] = self.df_database['GF'].astype(int)
        self.df_database['GA'] = self.df_database['GA'].astype(int)

        self.df_database = self.df_database.drop_duplicates()

        if current_year == True:
            self.df_database.to_csv("dataset\cleaned\dataset_cleaned_current_year.csv")
        else:
            self.df_database.to_csv("dataset\cleaned\dataset_cleaned.csv")
        print("Dataset cleaned and saved on dataset\cleaned")

