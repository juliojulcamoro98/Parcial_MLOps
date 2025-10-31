
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------- Parámetros --------
DATA_PATH   = r"C:\Users\julio\Documents\Mlops\mlops_project\data\raw\Data_CU_venta.csv"
TARGET_COL  = "target"
PERIOD_COL  = "p_codmes"
MISSING_TH  = 0.80                
RANDOM_SEED = 42

# -------- Utilidades --------
def load_csv_flexible(file_path: str) -> pd.DataFrame:
    """Carga flexible de CSV (delimitador auto y tolerante a líneas malas)."""
    return pd.read_csv(
        file_path,
        engine="python",
        sep=None,
        on_bad_lines="skip",
        quotechar='"',
        doublequote=True,
        escapechar="\\",
        encoding="utf-8-sig"
    )

def drop_columns_by_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Elimina columnas con fracción de NaN > threshold (0-1)."""
    missing_fraction = df.isna().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
    return df.drop(columns=cols_to_drop)

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa faltantes:
      - Categóricas (object/category): moda
      - Numéricas: mediana
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object" or str(out[c].dtype) == "category":
            if out[c].isna().any():
                mode_val = out[c].mode(dropna=True)
                if not mode_val.empty:
                    out[c] = out[c].fillna(mode_val[0])
                else:
                    out[c] = out[c].fillna("NA")
        else:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].median())
    return out

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia nombres de columnas:
      - minúsculas, no alfa-num -> "_", trim "_"
      - asegura unicidad agregando sufijos _1, _2...
    """
    out = df.copy()
    raw = [str(c).lower() for c in out.columns]
    base = [re.sub(r"[^\w]+", "_", c).strip("_") for c in raw]
    seen = {}
    final_cols = []
    for c in base:
        if c not in seen:
            seen[c] = 0
            final_cols.append(c)
        else:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
    out.columns = final_cols
    return out

def encode_categoricals(df: pd.DataFrame, suffix="_enc") -> pd.DataFrame:
    """
    Codifica *todas* las columnas categóricas/texto con LabelEncoder,
    agregando nuevas columnas con sufijo.
    """
    out = df.copy()
    cat_cols = [c for c in out.columns if out[c].dtype == "object" or str(out[c].dtype) == "category"]
    for c in cat_cols:
        le = LabelEncoder()
        out[c + suffix] = le.fit_transform(out[c].astype(str))
    return out

df = load_csv_flexible(DATA_PATH)

if PERIOD_COL in df.columns:
    if df[PERIOD_COL].isna().any():
        moda_periodo = df[PERIOD_COL].mode(dropna=True)
        if not moda_periodo.empty:
            df[PERIOD_COL] = df[PERIOD_COL].fillna(moda_periodo[0])
    df[PERIOD_COL] = pd.to_numeric(df[PERIOD_COL], errors="coerce").astype("Int64")

df = drop_columns_by_missing(df, MISSING_TH)

df = impute_missing(df)

df = clean_column_names(df)

df = encode_categoricals(df, suffix="_enc")

if TARGET_COL in df.columns:
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
else:
    raise ValueError(f"No se encontró la columna objetivo '{TARGET_COL}'.")

num_like = df.select_dtypes(include=["number", "boolean"]).columns.tolist()
feature_cols = [c for c in num_like if c != TARGET_COL]
for c in df[feature_cols].select_dtypes(include=["boolean"]).columns:
    df[c] = df[c].astype("int8")

if PERIOD_COL not in df.columns:
    raise ValueError(f"No se encontró la columna de periodo '{PERIOD_COL}' para el split estratificado.")

last_period = int(pd.Series(df[PERIOD_COL].dropna()).max())
df_hist = df[df[PERIOD_COL] != last_period].copy()
df_test = df[df[PERIOD_COL] == last_period].copy()

if df_hist.empty or df_test.empty:
    raise ValueError("Revisa 'p_codmes': no hay historial suficiente o no existe un último periodo para test.")

X_hist = df_hist[feature_cols].copy()
y_hist = df_hist[TARGET_COL].copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X_hist, y_hist,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df_hist[PERIOD_COL]  
)

X_test = df_test[feature_cols].copy()
y_test = df_test[TARGET_COL].copy()

print(f"Último periodo (test): {last_period}")
print("Distribución por periodo en histórico (antes del split):")
print(df_hist[PERIOD_COL].value_counts().sort_index())

print("\nTamaños ->",
      f"train: {X_train.shape}, valid: {X_valid.shape}, test: {X_test.shape}")


# -------- Exportar splits a CSV --------
OUTPUT_DIR = r"C:\Users\julio\Documents\Mlops\mlops_project\data\processed"

X_train.assign(target=y_train).to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
X_valid.assign(target=y_valid).to_csv(f"{OUTPUT_DIR}/valid.csv", index=False)
X_test.assign(target=y_test).to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print("\n✅ Archivos CSV exportados:")
print(f"{OUTPUT_DIR}/train.csv")
print(f"{OUTPUT_DIR}/valid.csv")
print(f"{OUTPUT_DIR}/test.csv")