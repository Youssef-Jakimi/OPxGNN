"""Prétraitement du dataset Electricity Load Diagrams 2011–2014.

Objectifs (cf. project_context.md) :
- Charger le fichier texte séparé par ';'
- Re-échantillonner de 15 min à 1 h (moyenne)
- Gérer les clients "créés" après 2011 (valeurs initiales à 0)
- Générer des labels de classification (Normal=0 / Congestionné=1)
- Split strictement chronologique (train/val/test)

Sorties :
- data/processed/load_hourly.csv.gz
- data/processed/labels_hourly.csv.gz
- data/processed/thresholds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {path}. "
            "Placez LD2011_2014.txt dans le dossier dataset/."
        )

    # Le fichier UCI est un CSV-like séparé par ';'.
    # Certaines versions ont un ';' final => colonne vide à supprimer.
    df = pd.read_csv(
        path,
        sep=";",
        header=0,
        index_col=0,
        parse_dates=[0],
        engine="python",
    )

    # Supprime les colonnes entièrement vides (ex: colonne créée par ';' final)
    df = df.dropna(axis=1, how="all")

    # Assure un index datetime trié
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    # Conversion robuste en numérique
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _mask_pre_creation_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace par NaN les zéros initiaux (avant la première valeur non nulle).

    Hypothèse : pour certains clients, les valeurs initiales à 0 signifient
    "client non encore présent" plutôt que "consommation réellement nulle".

    Note : on ne remplace PAS les zéros après l'apparition du client.
    """

    def mask_series(s: pd.Series) -> pd.Series:
        nonzero = (s.notna()) & (s.to_numpy() != 0)
        if not bool(nonzero.any()):
            return s
        first = int(np.flatnonzero(nonzero.to_numpy())[0])
        if first > 0:
            s = s.copy()
            s.iloc[:first] = np.nan
        return s

    return df.apply(mask_series, axis=0)


def _resample_hourly(df_15m: pd.DataFrame) -> pd.DataFrame:
    # Moyenne horaire (kW) : cohérent pour lisser les anomalies DST.
    df_h = df_15m.resample("1h").mean()
    return df_h


def _make_splits(
    index: pd.DatetimeIndex,
    train_end: str,
    val_end: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split strictement chronologique.

    Convention :
    - train : <= train_end
    - val   : (train_end, val_end]
    - test  : > val_end
    """

    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_mask = index <= train_end_dt
    val_mask = (index > train_end_dt) & (index <= val_end_dt)
    test_mask = index > val_end_dt

    return (
        pd.Series(train_mask, index=index),
        pd.Series(val_mask, index=index),
        pd.Series(test_mask, index=index),
    )


def _compute_thresholds(
    df_h: pd.DataFrame,
    train_mask: pd.Series,
    scheme: str,
    q: float,
) -> pd.Series:
    train = df_h.loc[train_mask.values]

    if scheme == "per_node_quantile":
        # Un seuil par client (quantile q sur la période train).
        return train.quantile(q=q, axis=0, numeric_only=True)

    if scheme == "global_from_node_max":
        # Un seuil global basé sur le quantile q des maxima par client (sur train).
        node_max = train.max(axis=0, numeric_only=True)
        global_thr = float(node_max.quantile(q=q))
        return pd.Series(global_thr, index=df_h.columns)

    raise ValueError(
        "Schéma de label inconnu. Choix: per_node_quantile, global_from_node_max"
    )


def _make_labels(df_h: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    # 1 si conso > seuil, sinon 0
    thr = thresholds.reindex(df_h.columns)
    labels = (df_h.gt(thr, axis=1)).astype("int8")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prétraitement + labels pour Electricity Load Diagrams 2011-2014"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset") / "LD2011_2014.txt",
        help="Chemin vers LD2011_2014.txt",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data") / "processed",
        help="Dossier de sortie",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2013-12-31 23:00:00",
        help="Fin de la période train (incluse)",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2014-06-30 23:00:00",
        help="Fin de la période validation (incluse)",
    )
    parser.add_argument(
        "--label-scheme",
        type=str,
        default="per_node_quantile",
        choices=["per_node_quantile", "global_from_node_max"],
        help="Schéma de seuil pour les labels",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile utilisé pour le seuil (ex: 0.95)",
    )
    parser.add_argument(
        "--keep-nans",
        action="store_true",
        help="Conserver les NaN dans les sorties (sinon remplissage à 0)",
    )

    args = parser.parse_args()

    df = _load_raw(args.input)
    df_h = _resample_hourly(df)
    df_h = _mask_pre_creation_zeros(df_h)

    train_mask, val_mask, test_mask = _make_splits(df_h.index, args.train_end, args.val_end)

    thresholds = _compute_thresholds(
        df_h=df_h,
        train_mask=train_mask,
        scheme=args.label_scheme,
        q=args.quantile,
    )
    labels = _make_labels(df_h, thresholds)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.keep_nans:
        df_h = df_h.fillna(0.0)
        labels = labels.fillna(0)

    df_h.to_csv(outdir / "load_hourly.csv.gz", compression="gzip")
    labels.to_csv(outdir / "labels_hourly.csv.gz", compression="gzip")
    thresholds.rename("threshold").to_csv(outdir / "thresholds.csv")

    # Petit récapitulatif (utile en équipe)
    n_total = len(df_h.index)
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    n_test = int(test_mask.sum())

    print("Prétraitement terminé.")
    print(f"- Période: {df_h.index.min()} -> {df_h.index.max()} ({n_total} pas horaires)")
    print(f"- Split: train={n_train}, val={n_val}, test={n_test}")
    print(f"- Sorties: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
