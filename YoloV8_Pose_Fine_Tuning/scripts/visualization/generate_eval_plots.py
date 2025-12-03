#!/usr/bin/env python3
"""Genera gráficos de evaluación a partir de los CSVs en outputs/runs/<run>.

Genera:
- oks_histogram.png  -> histograma de `oks_mean` utilizando `custom_metrics.csv` (distribución por época)
- pck_by_keypoint.png -> barra / tabla con PCK@0.1 por keypoint usando `final_evaluation_metrics.csv`

Uso:
    python scripts/visualization/generate_eval_plots.py --run salmon_pose_v124

"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap


def read_final_metrics(csv_path: Path):
    df = pd.read_csv(csv_path, index_col=0)
    # The file format from the repo is with numbers in column '0'
    if '0' in df.columns:
        s = df['0']
    else:
        # fallback: take first column
        s = df.iloc[:, 0]
    return s


def make_oks_histogram(custom_csv: Path, out_path: Path):
    df = pd.read_csv(custom_csv)
    if 'oks_mean' not in df.columns:
        print(f"No se encontró 'oks_mean' en {custom_csv}. Columnas: {df.columns}")
        return False

    oks = df['oks_mean'].dropna().astype(float)
    plt.figure(figsize=(6,4))
    plt.hist(oks, bins=25, color='#2a9d8f', edgecolor='black')
    plt.title('Distribución de OKS (por época)')
    plt.xlabel('OKS')
    plt.ylabel('Frecuencia')
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Guardado histograma OKS en: {out_path}")
    return True


def make_pck_bar(final_csv: Path, out_path: Path, threshold='0.1'):
    s = read_final_metrics(final_csv)
    # Filtrar claves pck@{threshold}_*
    prefix = f'pck@{threshold}_'
    keys = [k for k in s.index if k.startswith(prefix)]
    if not keys:
        print(f"No se encontraron claves con prefijo {prefix} en {final_csv}")
        return False

    parts = [k.replace(prefix, '') for k in keys]
    values = [float(s[k]) for k in keys]

    # Crear figura tipo tabla/bar
    fig, ax = plt.subplots(figsize=(8, max(2, len(parts)*0.25+1)))
    y_pos = np.arange(len(parts))
    ax.barh(y_pos, values, color='#264653')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(parts)
    ax.invert_yaxis()
    ax.set_xlabel(f'PCK@{threshold} (%)')
    ax.set_title(f'PCK@{threshold} por Keypoint')
    for i, v in enumerate(values):
        ax.text(v + 0.5, i + 0.1, f"{v:.2f}%", color='black', fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Guardado gráfico PCK por keypoint en: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='salmon_pose_v124', help='Nombre del run en outputs/runs')
    args = parser.parse_args()

    base = Path(__file__).parents[2] / 'outputs' / 'runs' / args.run
    if not base.exists():
        print(f"Run no encontrada: {base}")
        return 2

    custom_csv = base / 'custom_metrics.csv'
    final_csv = base / 'final_evaluation_metrics.csv'

    oks_out = base / 'oks_histogram.png'
    pck_out = base / 'pck_by_keypoint.png'

    ok1 = False
    ok2 = False
    if custom_csv.exists():
        ok1 = make_oks_histogram(custom_csv, oks_out)
    else:
        print(f"No existe {custom_csv} — se omitirá histograma OKS")

    if final_csv.exists():
        ok2 = make_pck_bar(final_csv, pck_out, threshold='0.1')
    else:
        print(f"No existe {final_csv} — se omitirá gráfico PCK")

    if ok1 and ok2:
        print("Generación completada correctamente.")
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
