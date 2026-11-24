import numpy as np
import pandas as pd

# 1. Carga directa desde los archivos .xpt (o desde las URLs oficiales del CDC)

#Mediciones biologicas: Contiene mediciones de metales en sangre, entre ellos plomo, cadmio y mercurio.
url_pb = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PBCD_G.xpt"

#Evaluación cognitiva: Contiene resultados de pruebas cognitivas y de memoria, incluyendo CERAD, Animal Fluency y Digit Symbol Coding.
url_cfq = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/CFQ_G.xpt"

#Demografía: Contiene información demográfica de los participantes, incluyendo edad, sexo, nivel educativo y nivel de ingresos.
url_demo = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt"
)


# Carga de los datasets en DataFrames de pandas
df_pb = pd.read_sas(url_pb)
df_cfq = pd.read_sas(url_cfq)
df_demo = pd.read_sas(url_demo)

# 2. Selección de columnas relevantes antes del cruce

#mediciones biologicas: SEQN (identificador único del participante), LBXBPB (plomo en sangre), LBDBPBLC (plomo en sangre corregido), LBXBCD (cadmio en sangre), LBXTHG (mercurio en sangre)
cols_pb = ["SEQN", "LBXBPB", "LBDBPBLC", "LBXBCD", "LBXTHG"]

#evaluación cognitiva: SEQN (identificador único del participante), CFDCST1, CFDCST2, CFDCST3 (pruebas de memoria inmediata CERAD), CFDCSR (prueba de fluidez verbal Animal Fluency), CFDAST (prueba de codificación de dígitos Digit Symbol Coding), CFDDS (prueba de memoria diferida CERAD)
cols_cfq = [
    "SEQN",
    "CFDCST1",
    "CFDCST2",
    "CFDCST3",
    "CFDCSR",
    "CFDAST",
    "CFDDS",
]

#demografía: SEQN (identificador único del participante), RIDAGEYR (edad), RIAGENDR (sexo), DMDEDUC2 (nivel educativo), INDFMPIR (nivel de ingresos), RIDRETH3 (origen étnico)
cols_demo = ["SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2", "INDFMPIR", "RIDRETH3"]

# 3. Unión (Inner Join) por SEQN esto garantiza que solo se mantengan los participantes que tienen datos completos en las tres tablas
df = (
    df_demo[cols_demo]
    .merge(df_pb[cols_pb], on="SEQN", how="inner")
    .merge(df_cfq[cols_cfq], on="SEQN", how="inner")
)

# 4. Limpieza de códigos especiales de NHANES y valores nulos
# En DMDEDUC2, 7 (Refused) y 9 (Don't know) son valores faltantes reales
df["DMDEDUC2"] = df["DMDEDUC2"].replace({7: np.nan, 9: np.nan})

# Eliminar filas sin medición de plomo o sin las pruebas cognitivas principales
df = df.dropna(subset=["LBXBPB", "CFDDS", "CFDAST", "CFDCSR", "DMDEDUC2"]).copy()

# 5. Ingeniería de Características (Feature Engineering)
# Puntaje acumulado de aprendizaje inmediato CERAD (máx 30 puntos)
df["CERAD_INMEDIATO"] = df["CFDCST1"] + df["CFDCST2"] + df["CFDCST3"]

# Etiquetas legibles para análisis exploratorio
df["SEXO"] = df["RIAGENDR"].map({1: "Hombre", 2: "Mujer"})
mapa_edu = {
    1: "Menor a 9no grado",
    2: "Secundaria incompleta",
    3: "Secundaria completa",
    4: "Universidad incompleta",
    5: "Universitario graduado",
}
df["NIVEL_EDUCATIVO"] = df["DMDEDUC2"].map(mapa_edu)

# Cuartiles de plomo en sangre (ideal para comparar grupos de exposición en adultos)
df["CUARTIL_PLOMO"] = pd.qcut(
    df["LBXBPB"],
    q=4,
    labels=["Q1 (Bajo)", "Q2 (Medio-Bajo)", "Q3 (Medio-Alto)", "Q4 (Alto)"],
)

# 6. Verificación rápida de resultados
print(f"Dataset maestro listo: {df.shape[0]} pacientes y {df.shape[1]} variables.")
print("\nPromedio de pruebas cognitivas por cuartil de plomo en sangre:")
resumen = df.groupby("CUARTIL_PLOMO", observed=False)[
    ["LBXBPB", "RIDAGEYR", "CERAD_INMEDIATO", "CFDCSR", "CFDAST", "CFDDS"]
].mean()
print(resumen.round(2))

# Guardar dataset procesado
df.to_csv("nhanes_plomo_cognicion_2011.csv", index=False)