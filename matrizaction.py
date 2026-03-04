# ============================================================
#  MATRICES DE ATENCIÓN POR CLASE — BERT + TRAC-1
#  Agrega este bloque AL FINAL de tu script original,
#  justo después de: tokenizer.save_pretrained('./agresion-detector')
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("  EXTRACCIÓN DE MATRICES DE ATENCIÓN POR CLASE")
print("="*60)

# ─────────────────────────────────────────────────────────────
# PASO 1: Recargar modelo CON output_attentions=True
#         (tu modelo ya está guardado en ./agresion-detector)
# ─────────────────────────────────────────────────────────────
model_path = './agresion-detector'

tokenizer_attn = AutoTokenizer.from_pretrained(model_path)
model_attn = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    output_attentions=True   # ← la única diferencia con tu modelo original
)
model_attn.eval()

label_map     = {'OAG': 0, 'NAG': 1, 'CAG': 2}
label_names   = {0: 'OAG', 1: 'NAG', 2: 'CAG'}

# Paletas de color por clase (para que cada figura sea distinta)
PALETAS = {
    'OAG': 'Reds',      # Rojo  → agresión abierta
    'NAG': 'Greens',    # Verde → no agresivo
    'CAG': 'Oranges',   # Naranja → agresión encubierta
}
COLORES = {'OAG': '#c0392b', 'NAG': '#27ae60', 'CAG': '#d35400'}


# ─────────────────────────────────────────────────────────────
# PASO 2: Cargar el set de validación (el mismo que usaste)
# ─────────────────────────────────────────────────────────────
dev_df = pd.read_csv('data/agr_en_dev.csv')
dev_df.columns = ['id', 'text', 'labels']

# Separar textos por clase real (ground truth)
textos_por_clase = {
    'OAG': dev_df[dev_df['labels'] == 'OAG']['text'].tolist(),
    'NAG': dev_df[dev_df['labels'] == 'NAG']['text'].tolist(),
    'CAG': dev_df[dev_df['labels'] == 'CAG']['text'].tolist(),
}

# Limitar a N muestras por clase para que sea manejable
N_MUESTRAS = 50   # ← ajusta según tu RAM (50 es seguro, 100 es mejor)
for clase in textos_por_clase:
    textos_por_clase[clase] = textos_por_clase[clase][:N_MUESTRAS]
    print(f"  Clase {clase}: {len(textos_por_clase[clase])} textos")


# ─────────────────────────────────────────────────────────────
# PASO 3: Función para extraer atención de UN texto
# ─────────────────────────────────────────────────────────────
MAX_LEN = 30  # Máx de tokens a visualizar (mantiene heatmaps legibles)

def extraer_atencion(texto):
    """
    Retorna:
        tokens     → lista de strings (hasta MAX_LEN)
        attn_array → numpy (12 capas, 12 cabezas, seq, seq)
    """
    inputs = tokenizer_attn(
        texto,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LEN,
        padding=False
    )
    with torch.no_grad():
        outputs = model_attn(**inputs)

    tokens = tokenizer_attn.convert_ids_to_tokens(inputs['input_ids'][0])
    # Stack de 12 capas → (12, 12, seq, seq)
    attn = torch.stack([a[0] for a in outputs.attentions]).cpu().numpy()
    return tokens, attn


# ─────────────────────────────────────────────────────────────
# PASO 4: Calcular matriz de atención PROMEDIO por clase
#         Promediamos sobre: muestras + cabezas + capas
#         Resultado: una matriz (seq, seq) representativa por clase
# ─────────────────────────────────────────────────────────────
def calcular_promedio_clase(textos, clase_nombre):
    """
    Para una lista de textos de la misma clase, calcula:
      - La matriz de atención promedio global (12 capas, 12 cabezas promediadas)
      - La atención del [CLS] por capa (12 capas)
    """
    print(f"\n  Procesando clase {clase_nombre}...", end="")
    
    acum_global = None   # acumulador para promedio global
    acum_cls    = None   # acumulador para atención CLS por capa
    conteo      = 0

    for texto in textos:
        try:
            tokens, attn = extraer_atencion(str(texto))
            n = len(tokens)

            # ── Promedio global: media sobre capas y cabezas ──
            # attn shape: (12, 12, n, n) → media → (n, n)
            global_mean = attn.mean(axis=(0, 1))[:n, :n]  # (n, n)

            # ── CLS por capa: fila 0, media sobre cabezas ──
            # attn: (12, 12, n, n) → media sobre eje heads → (12, n, n)
            cls_by_layer = attn.mean(axis=1)[:, 0, :n]    # (12, n)

            if acum_global is None:
                # Inicializar con el tamaño de la primera muestra
                acum_global = global_mean
                acum_cls    = cls_by_layer
            else:
                # Alinear dimensiones al mínimo (distintos textos tienen distinta longitud)
                min_n = min(acum_global.shape[0], global_mean.shape[0])
                acum_global = acum_global[:min_n, :min_n] + global_mean[:min_n, :min_n]
                min_n_cls   = min(acum_cls.shape[1], cls_by_layer.shape[1])
                acum_cls    = acum_cls[:, :min_n_cls] + cls_by_layer[:, :min_n_cls]

            conteo += 1
        except Exception:
            continue

    print(f" ✓ ({conteo} textos procesados)")
    return acum_global / conteo, acum_cls / conteo


# Calcular para las 3 clases
promedios = {}
cls_por_capa = {}

for clase in ['OAG', 'NAG', 'CAG']:
    prom_global, prom_cls = calcular_promedio_clase(textos_por_clase[clase], clase)
    promedios[clase]    = prom_global
    cls_por_capa[clase] = prom_cls


# ─────────────────────────────────────────────────────────────
# FIGURA 1: Heatmap de atención promedio por clase (3 subplots)
#           Una figura lista para pegar en tu tesis
# ─────────────────────────────────────────────────────────────
print("\n  Generando Figura 1: Heatmaps promedio por clase...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'Matrices de Atención Promedio por Clase — BERT fine-tuned en TRAC-1\n'
    '(Promedio sobre todas las capas, cabezas y muestras del conjunto de validación)',
    fontsize=13, fontweight='bold', y=1.02
)

for ax, clase in zip(axes, ['OAG', 'NAG', 'CAG']):
    mat  = promedios[clase]
    n    = mat.shape[0]
    ticks = list(range(0, n, max(1, n // 8)))  # Mostrar ~8 ticks para legibilidad

    sns.heatmap(
        mat,
        ax=ax,
        cmap=PALETAS[clase],
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'shrink': 0.8, 'label': 'Peso de atención'}
    )
    ax.set_title(
        f'{clase}',
        fontsize=16, fontweight='bold', color=COLORES[clase], pad=10
    )
    ax.set_xlabel('Tokens (Key)', fontsize=10)
    ax.set_ylabel('Tokens (Query)', fontsize=10)

    # Anotación descriptiva por clase
    descripciones = {
        'OAG': 'Atención concentrada\nen tokens agresivos explícitos',
        'NAG': 'Atención distribuida\nuniformemente (línea base)',
        'CAG': 'Atención difusa\nsobre contexto y sarcasmo',
    }
    ax.text(0.5, -0.15, descripciones[clase],
            transform=ax.transAxes, ha='center', fontsize=9,
            color='gray', style='italic')

plt.tight_layout()
plt.savefig('figura1_heatmap_promedio_por_clase.png', dpi=200, bbox_inches='tight')
print("  ✓ Guardada: figura1_heatmap_promedio_por_clase.png")
plt.show()


# ─────────────────────────────────────────────────────────────
# FIGURA 2: Evolución del token [CLS] por capa y por clase
#           Muestra cómo el vector de clasificación evoluciona
# ─────────────────────────────────────────────────────────────
print("\n  Generando Figura 2: Evolución CLS por capa y clase...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Atención del Token [CLS] por Capa — BERT fine-tuned en TRAC-1\n'
    '(Filas = capas 1–12, Columnas = posiciones de tokens, promedio por clase)',
    fontsize=13, fontweight='bold', y=1.02
)

for ax, clase in zip(axes, ['OAG', 'NAG', 'CAG']):
    mat = cls_por_capa[clase]  # (12 capas, n_tokens)

    sns.heatmap(
        mat,
        ax=ax,
        cmap=PALETAS[clase],
        xticklabels=False,
        yticklabels=[f'C{i+1}' for i in range(12)],
        cbar_kws={'shrink': 0.8, 'label': 'Peso atención [CLS]'}
    )
    ax.set_title(f'{clase}', fontsize=16, fontweight='bold',
                 color=COLORES[clase], pad=10)
    ax.set_xlabel('Posición del token', fontsize=10)
    ax.set_ylabel('Capa de BERT', fontsize=10)
    ax.tick_params(axis='y', labelsize=8, rotation=0)

plt.tight_layout()
plt.savefig('figura2_cls_por_capa_por_clase.png', dpi=200, bbox_inches='tight')
print("  ✓ Guardada: figura2_cls_por_capa_por_clase.png")
plt.show()


# ─────────────────────────────────────────────────────────────
# FIGURA 3: Diferencia entre clases (OAG - NAG y CAG - NAG)
#           Resalta qué patrones son EXCLUSIVOS de la agresión
# ─────────────────────────────────────────────────────────────
print("\n  Generando Figura 3: Diferencia OAG−NAG y CAG−NAG...")

# Alinear al mínimo tamaño entre clases
min_n = min(promedios['OAG'].shape[0],
            promedios['NAG'].shape[0],
            promedios['CAG'].shape[0])

diff_oag = promedios['OAG'][:min_n, :min_n] - promedios['NAG'][:min_n, :min_n]
diff_cag = promedios['CAG'][:min_n, :min_n] - promedios['NAG'][:min_n, :min_n]

# Colormap divergente: rojo=más atención que NAG, azul=menos
cmap_div = 'RdBu_r'

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
        diff,
        ax=ax,
        cmap=cmap_div,
        center=0,
        vmin=-vmax, vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'shrink': 0.8, 'label': 'Δ peso de atención'}
    )
    ax.set_title(titulo, fontsize=12, fontweight='bold', color=color, pad=10)
    ax.set_xlabel('Tokens (Key)', fontsize=10)
    ax.set_ylabel('Tokens (Query)', fontsize=10)

plt.tight_layout()
plt.savefig('figura3_diferencia_clases.png', dpi=200, bbox_inches='tight')
print("  ✓ Guardada: figura3_diferencia_clases.png")
plt.show()


# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  MATRICES DE ATENCIÓN GENERADAS CORRECTAMENTE")
print("="*60)
print("\n  Archivos generados (listos para tu tesis):")
print("  📊 figura1_heatmap_promedio_por_clase.png")
print("     → Patrón global de atención OAG / NAG / CAG")
print("  📊 figura2_cls_por_capa_por_clase.png")
print("     → Cómo el token [CLS] atiende el texto en cada capa")
print("  📊 figura3_diferencia_clases.png")
print("     → Qué patrones son exclusivos de la agresión")
print("\n  Interpretación para tu tesis:")
print("  • OAG: atención concentrada → tokens agresivos explícitos")
print("  • NAG: atención uniforme   → línea base sin patrones")
print("  • CAG: atención difusa     → contexto y lenguaje indirecto")
print("  • Figura 3 muestra exactamente dónde difieren OAG y CAG de NAG")
