"""Prétraitement du dataset Electricity Load Diagrams 2011–2014.

Pipeline :
- Charger le fichier texte séparé par ';' (décimales ',')
- Ré-échantillonner de 15 min à 1 h (moyenne) + fillna
- Variable cible : charge totale, seuil 95e centile (train), labels 0/1
- Split stratifié (train/val/test) pour conserver les classes dans chaque split
- Normalisation MinMax (fit sur train uniquement)
- Séquences temporelles (fenêtre 24 h à pas horaire)
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Chemins et hyperparamètres
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
n_steps = 24  # 24 h de contexte à résolution horaire
CONGESTION_PERCENTILE = 95
SPLIT_RANDOM_STATE = 42

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sequences-only",
    action="store_true",
    help="Reprendre depuis data/processed/load_hourly.csv.gz",
)
parser.add_argument(
    "--export-x-csv",
    action="store_true",
    help="Exporter aussi X_*.csv aplatis (très volumineux)",
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Ne pas régénérer les fichiers sequences_*.npz déjà présents",
)
args = parser.parse_args()


def resolve_work_dir() -> Path:
    cwd = Path.cwd()
    if (cwd / "LD2011_2014.txt").exists():
        return cwd
    candidate = PROJECT_ROOT / "dataset" / "dataelectricity"
    if (candidate / "LD2011_2014.txt").exists():
        return candidate
    return cwd


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Tâche E : 15 min -> 1 h (moyenne), puis fillna."""
    datetime_col = df.columns[0]
    df = df.rename(columns={datetime_col: "datetime"})
    client_cols = [c for c in df.columns if c != "datetime"]

    df = df.fillna(0)
    df = df.set_index("datetime")
    missing_before = int(df.isnull().sum().sum())

    df_hourly = df.resample("1h").mean()
    missing_after_resample = int(df_hourly.isnull().sum().sum())
    df_hourly = df_hourly.fillna(0)

    df_hourly = df_hourly.reset_index()
    print(f"  Lignes 15 min : {len(df)} -> horaire : {len(df_hourly)}")
    print(f"  NaN avant resample : {missing_before}, après resample : {missing_after_resample}")
    return df_hourly, client_cols


def build_classification_labels(
    df_hourly: pd.DataFrame, client_cols: list[str], train_end: int
) -> tuple[pd.Series, float]:
    """Tâche F : charge totale + seuil 95e centile (train) -> labels 0/1."""
    load_total = df_hourly[client_cols].sum(axis=1)
    threshold = float(np.percentile(load_total.iloc[:train_end], CONGESTION_PERCENTILE))
    labels = (load_total > threshold).astype(np.int8)
    n_congested = int(labels.sum())
    print(f"  Charge totale (kW) : min={load_total.min():.1f}, max={load_total.max():.1f}")
    print(f"  Seuil ({CONGESTION_PERCENTILE}e centile, train) : {threshold:.2f} kW")
    print(
        f"  Labels : {n_congested} congestés (1) / {len(labels) - n_congested} normaux (0) "
        f"({100 * n_congested / len(labels):.2f} % en congestion)"
    )
    return load_total, threshold, labels


def class_counts(labels: np.ndarray | pd.Series) -> dict[int, int]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    values, counts = np.unique(labels_arr, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def print_class_distribution(name: str, labels: np.ndarray | pd.Series) -> None:
    counts = class_counts(labels)
    print(f"{name.upper()}:")
    print(f"0={counts.get(0, 0)}")
    print(f"1={counts.get(1, 0)}")


def assert_binary_split(name: str, labels: np.ndarray | pd.Series) -> None:
    counts = class_counts(labels)
    if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
        raise ValueError(
            f"Split {name} invalide: il doit contenir les deux classes, "
            f"recu {counts}."
        )


def validate_split_distributions(split_labels: dict[str, np.ndarray | pd.Series]) -> None:
    for name, labels_part in split_labels.items():
        print_class_distribution(name, labels_part)
        assert_binary_split(name, labels_part)


def build_stratified_indices(labels_arr: np.ndarray) -> dict[str, np.ndarray]:
    labels_arr = np.asarray(labels_arr, dtype=np.int64)
    if not np.all(np.isin(labels_arr, [0, 1])):
        raise ValueError(f"Labels non binaires detectes: {class_counts(labels_arr)}")

    counts = class_counts(labels_arr)
    if len(counts) != 2:
        raise ValueError(f"Impossible de stratifier: une seule classe presente {counts}.")
    if min(counts.values()) < 3:
        raise ValueError(
            "Impossible de creer train/val/test stratifies: "
            f"classe minoritaire trop petite {counts}."
        )

    indices = np.arange(len(labels_arr))
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        stratify=labels_arr,
        random_state=SPLIT_RANDOM_STATE,
        shuffle=True,
    )

    relative_test_ratio = test_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        stratify=labels_arr[temp_idx],
        random_state=SPLIT_RANDOM_STATE,
        shuffle=True,
    )

    split_indices = {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }
    validate_split_distributions(
        {name: labels_arr[idx] for name, idx in split_indices.items()}
    )
    return split_indices


def save_processed_artifacts(
    df_hourly: pd.DataFrame,
    client_cols: list[str],
    load_total: pd.Series,
    labels: pd.Series,
    threshold: float,
    train_end: int,
    train_labels: pd.Series | np.ndarray | None = None,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    load_out = pd.concat(
        [df_hourly[["datetime"]], df_hourly[client_cols], load_total.rename("load_total_kw")],
        axis=1,
    )
    load_out.to_csv(PROCESSED_DIR / "load_hourly.csv.gz", index=False, compression="gzip")

    labels_out = pd.DataFrame(
        {"datetime": df_hourly["datetime"], "load_total_kw": load_total, "label": labels}
    )
    labels_out.to_csv(PROCESSED_DIR / "labels_hourly.csv.gz", index=False, compression="gzip")

    labels_for_train_stats = labels.iloc[:train_end] if train_labels is None else train_labels
    thresholds = pd.DataFrame(
        {
            "percentile": [CONGESTION_PERCENTILE],
            "threshold_kw": [threshold],
            "n_train_samples": [int(len(labels_for_train_stats))],
            "n_congested_train": [int(np.asarray(labels_for_train_stats).sum())],
            "threshold_calibration_samples": [train_end],
        }
    )
    thresholds.to_csv(PROCESSED_DIR / "thresholds.csv", index=False)

    print(f"  -> {PROCESSED_DIR / 'load_hourly.csv.gz'}")
    print(f"  -> {PROCESSED_DIR / 'labels_hourly.csv.gz'}")
    print(f"  -> {PROCESSED_DIR / 'thresholds.csv'}")


WORK_DIR = resolve_work_dir()
os.chdir(WORK_DIR)
print(f"Répertoire de travail : {WORK_DIR}")

missing_before = 0
df_normalized = None
hourly_split_indices = None

if args.sequences_only and (PROCESSED_DIR / "load_hourly.csv.gz").exists():
    print("\nChargement des données horaires traitées (mode sequences-only)...")
    load_df = pd.read_csv(PROCESSED_DIR / "load_hourly.csv.gz", parse_dates=["datetime"])
    labels_df = pd.read_csv(PROCESSED_DIR / "labels_hourly.csv.gz", parse_dates=["datetime"])
    client_cols = [
        c for c in load_df.columns if c not in ("datetime", "load_total_kw")
    ]
    df_hourly = load_df[["datetime"] + client_cols].copy()
    load_total = load_df["load_total_kw"]
    labels = labels_df["label"].astype(np.int8)
    df = df_hourly
    threshold = float(
        pd.read_csv(PROCESSED_DIR / "thresholds.csv")["threshold_kw"].iloc[0]
    )
else:
    file_path = WORK_DIR / "LD2011_2014.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {file_path}")

    print("\n" + "=" * 50)
    print("TÂCHE D — CHARGEMENT")
    print("=" * 50)
    df = pd.read_csv(
        file_path, sep=";", parse_dates=[0], low_memory=False, decimal=","
    )
    print(f"Dataset brut : {df.shape}")

    print("\n" + "=" * 50)
    print("TÂCHE E — RÉÉCHANTILLONNAGE 15 min -> 1 h")
    print("=" * 50)
    df_hourly, client_cols = resample_to_hourly(df)

    n = len(df_hourly)
    train_end = int(n * train_ratio)

    print("\n" + "=" * 50)
    print("TÂCHE F — LABELS DE CLASSIFICATION")
    print("=" * 50)
    load_total, threshold, labels = build_classification_labels(
        df_hourly, client_cols, train_end
    )

    print("\n" + "=" * 50)
    print("TÂCHE G — SPLIT STRATIFIÉ HORAIRE")
    print("=" * 50)
    hourly_split_indices = build_stratified_indices(labels.to_numpy(dtype=np.int8))

    print("\n" + "=" * 50)
    print("NORMALISATION (fit sur train uniquement)")
    print("=" * 50)
    scaler_minmax = MinMaxScaler()
    scaler_minmax.fit(df_hourly.iloc[hourly_split_indices["train"]][client_cols])
    data_normalized = scaler_minmax.transform(df_hourly[client_cols])
    print(
        f"  Plage brute clients : "
        f"[{df_hourly[client_cols].min().min():.2f}, {df_hourly[client_cols].max().max():.2f}] kW"
    )
    print(
        f"  Après MinMax : "
        f"[{data_normalized.min():.2f}, {data_normalized.max():.2f}]"
    )

    df_normalized = pd.DataFrame(data_normalized, columns=client_cols)
    df_normalized.insert(0, "datetime", df_hourly["datetime"].values)
    df_normalized["load_total_kw"] = load_total.values
    df_normalized["label"] = labels.values

    train_data = df_normalized.iloc[hourly_split_indices["train"]]
    val_data = df_normalized.iloc[hourly_split_indices["val"]]
    test_data = df_normalized.iloc[hourly_split_indices["test"]]
    print(f"  Total : {n} | Train : {len(train_data)} | Val : {len(val_data)} | Test : {len(test_data)}")

    print("\n" + "=" * 50)
    print("SAUVEGARDE")
    print("=" * 50)
    save_processed_artifacts(
        df_hourly,
        client_cols,
        load_total,
        labels,
        threshold,
        train_end,
        train_labels=labels.iloc[hourly_split_indices["train"]],
    )

    train_data.to_csv(WORK_DIR / "train_data.csv", index=False)
    val_data.to_csv(WORK_DIR / "validation_data.csv", index=False)
    test_data.to_csv(WORK_DIR / "test_data.csv", index=False)
    df_normalized.to_csv(WORK_DIR / "electricity_normalized_full.csv", index=False)
    joblib.dump(scaler_minmax, WORK_DIR / "scaler_minmax.pkl")
    print("  Fichiers locaux : train/validation/test_data.csv, electricity_normalized_full.csv")

    # Reconstruire df_normalized pour la suite si on vient du bloc else
    df = df_hourly

# Indices split (données horaires normalisées)
if df_normalized is None:
    scaler_path = WORK_DIR / "scaler_minmax.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            "scaler_minmax.pkl manquant. Relancez sans --sequences-only."
        )
    scaler_minmax = joblib.load(scaler_path)
    data_normalized = scaler_minmax.transform(df_hourly[client_cols])
    df_normalized = pd.DataFrame(data_normalized, columns=client_cols)
    df_normalized.insert(0, "datetime", df_hourly["datetime"].values)
    df_normalized["load_total_kw"] = load_total.values
    df_normalized["label"] = labels.values

n = len(df_normalized)
if hourly_split_indices is None:
    print("\n" + "=" * 50)
    print("SPLIT STRATIFIÉ HORAIRE")
    print("=" * 50)
    hourly_split_indices = build_stratified_indices(
        df_normalized["label"].to_numpy(dtype=np.int8)
    )

train_data = df_normalized.iloc[hourly_split_indices["train"]]
val_data = df_normalized.iloc[hourly_split_indices["val"]]
test_data = df_normalized.iloc[hourly_split_indices["test"]]

print("\n" + "=" * 50)
print("SÉQUENCES TEMPORELLES (fenêtre 24 h)")
print("=" * 50)

consumption_data = df_normalized[client_cols].to_numpy(dtype=np.float32)
labels_arr = df_normalized["label"].to_numpy(dtype=np.int8)
windows = sliding_window_view(consumption_data, window_shape=n_steps, axis=0)
n_samples = len(consumption_data) - n_steps
y_reg = np.ascontiguousarray(consumption_data[n_steps:])
y_cls = np.ascontiguousarray(labels_arr[n_steps:])

print(f"  Échantillons séquences : {n_samples}")
print(f"  X : (N, {n_steps}, {len(client_cols)}) | y_reg : {y_reg.shape} | y_cls : {y_cls.shape}")
print(f"  Labels séquences — congestés : {int(y_cls.sum())} ({100 * y_cls.mean():.2f} %)")


def save_flat_csv(path, array, chunk_size=512):
    n_rows, n_steps_local, n_clients = array.shape
    n_cols = n_steps_local * n_clients
    with open(path, "w", encoding="utf-8") as handle:
        for start in range(0, n_rows, chunk_size):
            chunk = array[start : start + chunk_size].reshape(-1, n_cols)
            np.savetxt(handle, chunk, delimiter=",", fmt="%.6f")


print("\n" + "=" * 50)
print("SPLIT STRATIFIÉ DES SÉQUENCES")
print("=" * 50)
sequence_split_indices = build_stratified_indices(y_cls)

print("\nExport des séquences :")
for name, split_idx in sequence_split_indices.items():
    npz_path = WORK_DIR / f"sequences_{name}.npz"
    y_csv_path = WORK_DIR / f"y_{name}.csv"
    y_cls_path = WORK_DIR / f"y_cls_{name}.csv"

    if args.skip_existing and npz_path.exists() and y_cls_path.exists():
        existing_labels = np.load(npz_path)["y_cls"]
        print(f"  {name} : ignoré (déjà présent)")
        print_class_distribution(name, existing_labels)
        assert_binary_split(name, existing_labels)
        continue

    X_part = np.ascontiguousarray(windows[split_idx].transpose(0, 2, 1))
    y_reg_part = y_reg[split_idx]
    y_cls_part = y_cls[split_idx]
    assert_binary_split(name, y_cls_part)
    print(f"  {name} : X={X_part.shape}, y_reg={y_reg_part.shape}, y_cls={y_cls_part.shape}")

    if not (args.skip_existing and npz_path.exists()):
        np.savez_compressed(
            npz_path, X=X_part, y=y_reg_part, y_cls=y_cls_part
        )
        print(f"    -> {npz_path.name} ({npz_path.stat().st_size / (1024 ** 2):.1f} MB)")

    if args.export_x_csv:
        save_flat_csv(WORK_DIR / f"X_{name}.csv", X_part)

    if not (args.skip_existing and y_csv_path.exists()):
        pd.DataFrame(y_reg_part, columns=client_cols).to_csv(y_csv_path, index=False)
    if not (args.skip_existing and y_cls_path.exists()):
        pd.DataFrame({"label": y_cls_part}).to_csv(y_cls_path, index=False)
        print(f"    -> {y_cls_path.name}")

    del X_part

print("\n" + "=" * 50)
print("RÉSUMÉ")
print("=" * 50)
print(f"  Résolution : 1 h | Fenêtre : {n_steps} h | Seuil congestion : {threshold:.2f} kW")
print(f"  Split : {len(train_data)} / {len(val_data)} / {len(test_data)}")
print(f"  Artefacts projet : {PROCESSED_DIR}")

print("\nFichiers générés :")
for path in [
    PROCESSED_DIR / "load_hourly.csv.gz",
    PROCESSED_DIR / "labels_hourly.csv.gz",
    PROCESSED_DIR / "thresholds.csv",
    WORK_DIR / "train_data.csv",
    WORK_DIR / "sequences_train.npz",
]:
    if path.exists():
        print(f"  - {path} ({path.stat().st_size / (1024 ** 2):.2f} MB)")
