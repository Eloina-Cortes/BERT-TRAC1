# ============================================================
#  matrizaction.py
#  Módulo de matrices de atención — BERT + TRAC-1
#  Uso desde main.py:
#      from matrizaction import generar_matrices_atencion
#      generar_matrices_atencion()
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generar_matrices_atencion(
    model_path='./agresion-detector',
    data_path='data/agr_en_dev.csv',
    n_muestras=50,
    max_len=30
):
    """
    Genera y guarda las 3 figuras de matrices de atención por clase.

    Parámetros:
        model_path  → ruta al modelo entrenado (default: './agresion-detector')
        data_path   → ruta al CSV de validación  (default: 'data/agr_en_dev.csv')
        n_muestras  → textos por clase a procesar (default: 50)
        max_len     → máx tokens por texto        (default: 30)

    Archivos generados:
        figura1_heatmap_promedio_por_clase.png
        figura2_cls_por_capa_por_clase.png
        figura3_diferencia_clases.png
    """

    print("\n" + "="*60)
    print("  EXTRACCIÓN DE MATRICES DE ATENCIÓN POR CLASE")
    print("="*60)

    # ── Constantes ───────────────────────────────────────────
    PALETAS = {'OAG': 'Reds', 'NAG': 'Greens', 'CAG': 'Oranges'}
    COLORES = {'OAG': '#c0392b', 'NAG': '#27ae60', 'CAG': '#d35400'}

    # ── PASO 1: Cargar modelo con output_attentions=True ─────
    print("\n  Cargando modelo con salida de atenciones...")
    tokenizer_attn = AutoTokenizer.from_pretrained(model_path)
    model_attn = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        output_attentions=True
    )
    model_attn.eval()
    print("  ✓ Modelo cargado")

    # ── PASO 2: Cargar datos de validación ───────────────────
    dev_df = pd.read_csv(data_path)
    dev_df.columns = ['id', 'text', 'labels']

    textos_por_clase = {
        'OAG': dev_df[dev_df['labels'] == 'OAG']['text'].tolist()[:n_muestras],
        'NAG': dev_df[dev_df['labels'] == 'NAG']['text'].tolist()[:n_muestras],
        'CAG': dev_df[dev_df['labels'] == 'CAG']['text'].tolist()[:n_muestras],
    }
    for clase, textos in textos_por_clase.items():
        print(f"  Clase {clase}: {len(textos)} textos")

    # ── PASO 3: Extraer atención de un texto ─────────────────
    def extraer_atencion(texto):
        inputs = tokenizer_attn(
            texto,
            return_tensors='pt',
            truncation=True,
            max_length=max_len,
            padding=False
        )
        with torch.no_grad():
            outputs = model_attn(**inputs)
        tokens = tokenizer_attn.convert_ids_to_tokens(inputs['input_ids'][0])
        attn = torch.stack([a[0] for a in outputs.attentions]).cpu().numpy()
        return tokens, attn

    # ── PASO 4: Calcular promedio por clase ──────────────────
    def calcular_promedio_clase(textos, clase_nombre):
        print(f"\n  Procesando clase {clase_nombre}...", end="")
        acum_global = None
        acum_cls    = None
        conteo      = 0

        for texto in textos:
            try:
                tokens, attn = extraer_atencion(str(texto))
                n = len(tokens)

                global_mean  = attn.mean(axis=(0, 1))[:n, :n]
                cls_by_layer = attn.mean(axis=1)[:, 0, :n]

                if acum_global is None:
                    acum_global = global_mean
                    acum_cls    = cls_by_layer
                else:
                    min_n     = min(acum_global.shape[0], global_mean.shape[0])
                    acum_global = acum_global[:min_n, :min_n] + global_mean[:min_n, :min_n]
                    min_n_cls = min(acum_cls.shape[1], cls_by_layer.shape[1])
                    acum_cls  = acum_cls[:, :min_n_cls] + cls_by_layer[:, :min_n_cls]

                conteo += 1
            except Exception:
                continue

        print(f" ✓ ({conteo} textos procesados)")
        return acum_global / conteo, acum_cls / conteo

    # Calcular para las 3 clases
    promedios    = {}
    cls_por_capa = {}
    for clase in ['OAG', 'NAG', 'CAG']:
        prom_global, prom_cls   = calcular_promedio_clase(textos_por_clase[clase], clase)
        promedios[clase]        = prom_global
        cls_por_capa[clase]     = prom_cls

    # ── FIGURA 1: Heatmap promedio por clase ─────────────────
    print("\n  Generando Figura 1: Heatmaps promedio por clase...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Matrices de Atención Promedio por Clase — BERT fine-tuned en TRAC-1\n'
        '(Promedio sobre todas las capas, cabezas y muestras del conjunto de validación)',
        fontsize=13, fontweight='bold', y=1.02
    )
    descripciones = {
        'OAG': 'Atención concentrada\nen tokens agresivos explícitos',
        'NAG': 'Atención distribuida\nuniformemente (línea base)',
        'CAG': 'Atención difusa\nsobre contexto y sarcasmo',
    }
    for ax, clase in zip(axes, ['OAG', 'NAG', 'CAG']):
        sns.heatmap(
            promedios[clase], ax=ax, cmap=PALETAS[clase],
            xticklabels=False, yticklabels=False,
            cbar_kws={'shrink': 0.8, 'label': 'Peso de atención'}
        )
        ax.set_title(clase, fontsize=16, fontweight='bold',
                     color=COLORES[clase], pad=10)
        ax.set_xlabel('Tokens (Key)', fontsize=10)
        ax.set_ylabel('Tokens (Query)', fontsize=10)
        ax.text(0.5, -0.15, descripciones[clase],
                transform=ax.transAxes, ha='center',
                fontsize=9, color='gray', style='italic')
    plt.tight_layout()
    plt.savefig('figura1_heatmap_promedio_por_clase.png', dpi=200, bbox_inches='tight')
    print("  ✓ Guardada: figura1_heatmap_promedio_por_clase.png")
    plt.show()

    # ── FIGURA 2: CLS por capa y clase ───────────────────────
    print("\n  Generando Figura 2: Evolución CLS por capa y clase...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Atención del Token [CLS] por Capa — BERT fine-tuned en TRAC-1\n'
        '(Filas = capas 1–12, Columnas = posiciones de tokens, promedio por clase)',
        fontsize=13, fontweight='bold', y=1.02
    )
    for ax, clase in zip(axes, ['OAG', 'NAG', 'CAG']):
        sns.heatmap(
            cls_por_capa[clase], ax=ax, cmap=PALETAS[clase],
            xticklabels=False,
            yticklabels=[f'C{i+1}' for i in range(12)],
            cbar_kws={'shrink': 0.8, 'label': 'Peso atención [CLS]'}
        )
        ax.set_title(clase, fontsize=16, fontweight='bold',
                     color=COLORES[clase], pad=10)
        ax.set_xlabel('Posición del token', fontsize=10)
        ax.set_ylabel('Capa de BERT', fontsize=10)
        ax.tick_params(axis='y', labelsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig('figura2_cls_por_capa_por_clase.png', dpi=200, bbox_inches='tight')
    print("  ✓ Guardada: figura2_cls_por_capa_por_clase.png")
    plt.show()

    # ── FIGURA 3: Diferencia entre clases ────────────────────
    print("\n  Generando Figura 3: Diferencia OAG−NAG y CAG−NAG...")
    min_n    = min(promedios['OAG'].shape[0],
                   promedios['NAG'].shape[0],
                   promedios['CAG'].shape[0])
    diff_oag = promedios['OAG'][:min_n, :min_n] - promedios['NAG'][:min_n, :min_n]
    diff_cag = promedios['CAG'][:min_n, :min_n] - promedios['NAG'][:min_n, :min_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Diferencia de Atención respecto a la Clase No Agresiva (NAG)\n'
        'Rojo = mayor atención que NAG  |  Azul = menor atención que NAG',
        fontsize=13, fontweight='bold', y=1.02
    )
    for ax, diff, titulo, color in zip(
        axes,
        [diff_oag, diff_cag],
        ['OAG − NAG\n(Agresión Abierta vs No Agresivo)',
         'CAG − NAG\n(Agresión Encubierta vs No Agresivo)'],
        ['#c0392b', '#d35400']
    ):
        vmax = np.abs(diff).max()
        sns.heatmap(
            diff, ax=ax, cmap='RdBu_r', center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=False, yticklabels=False,
            cbar_kws={'shrink': 0.8, 'label': 'Δ peso de atención'}
        )
        ax.set_title(titulo, fontsize=12, fontweight='bold', color=color, pad=10)
        ax.set_xlabel('Tokens (Key)', fontsize=10)
        ax.set_ylabel('Tokens (Query)', fontsize=10)
    plt.tight_layout()
    plt.savefig('figura3_diferencia_clases.png', dpi=200, bbox_inches='tight')
    print("  ✓ Guardada: figura3_diferencia_clases.png")
    plt.show()

    # ── Resumen ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  MATRICES DE ATENCIÓN GENERADAS CORRECTAMENTE")
    print("="*60)
    print("  📊 figura1_heatmap_promedio_por_clase.png")
    print("  📊 figura2_cls_por_capa_por_clase.png")
    print("  📊 figura3_diferencia_clases.png")
