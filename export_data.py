import pandas as pd
import lightkurve as lk
from General_Functions import pre_processed_curves

df_kepler = pre_processed_curves.open_datasets("KEPLER")

print(df_kepler.head())

# Filtrar "id_target" onde a "disposition" é "CONFIRMED"
confirmed_targets = df_kepler[df_kepler['disposition'] == 'CONFIRMED']['id_target'].sample(5).tolist()

# Filtrar "id_target" onde a "disposition" é "FALSE POSITIVE"
false_positive_targets = df_kepler[df_kepler['disposition'] == 'FALSE POSITIVE']['id_target'].sample(5).tolist()

# Combinar as listas
combined_targets = []
combined_targets.extend(confirmed_targets)
combined_targets.extend(false_positive_targets)

print("Combined Targets:", combined_targets)

data = []
for target in combined_targets:
    target = 'KIC ' + str(target)
    try:
        # Baixar dados brutos
        lc = lk.search_lightcurvefile(target).download().PDCSAP_FLUX
        
        # Converter para DataFrame
        df = lc.to_table().to_pandas()
        
        df["TARGET_KIC"] = target
        
        # Adicionar dados à lista
        data.append(df)
    except:
        continue
    
combined_data = pd.concat(data)

combined_data.to_csv('dados_brutos.csv', index=False)