#!/usr/bin/env python3
"""
Generate publication-quality 300 DPI comparative figures for Master Thesis:
Hyperspectral Image Classification (SS-ResNet INT8)
Platforms: AMD Kria KV260 (DPUCZDX8G) vs NVIDIA Jetson Orin Nano (TensorRT)
100% Real Empirical Data Measured on Physical Silicon.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for publication styling
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

output_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)

# Curated thesis palette
COLOR_KRIA = "#008080"    # Deep Teal (AMD Kria KV260)
COLOR_JETSON = "#76B900"  # NVIDIA Green (Jetson Orin Nano)
COLOR_ACCENT = "#2C3E50"  # Dark Slate

platforms = ["AMD Kria KV260", "NVIDIA Jetson Orin Nano"]
colors = [COLOR_KRIA, COLOR_JETSON]

# Exact empirical measurements from physical hardware
fps_values = [3057.97, 1787.91]
lat_patch_ms = [0.327, 0.559]
infer_scene_s = [3.486, 5.515]
total_scene_s = [6.376, 5.715]
pca_time_ms = [1240.51, 17.10]
power_w = [6.664, 6.074]
energy_patch_mj = [2.179, 3.397]
energy_scene_j = [42.490, 34.713]
oa_values = [98.30, 98.33]
aa_values = [97.60, 97.81]
kappa_values = [98.06, 98.10]

# =========================================================================
# 1. FIGURA 1: Rendimiento (Throughput en Silicio: FPS / Parches por Segundo)
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
bars = ax.bar(platforms, fps_values, color=colors, width=0.45, edgecolor="black", linewidth=1.2, zorder=3)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 60, f"{yval:,.1f} FPS", 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=COLOR_ACCENT)

ax.set_ylabel("Rendimiento (Parches / Segundo)", fontweight='bold')
ax.set_title("Rendimiento de Inferencia Hiperespectral (SS-ResNet INT8)\nDataset Indian Pines (9,225 Parches)", pad=15, fontweight='bold')
ax.set_ylim(0, 3600)
ax.axhline(0, color='black', linewidth=0.8)

speedup = fps_values[0] / fps_values[1]
ax.annotate(f"Kria KV260 es 1.71× más rápida en silicio\n(+71.0% Throughput)",
            xy=(0, 2550), xytext=(0.5, 3050),
            arrowprops=dict(facecolor=COLOR_KRIA, shrink=0.08, width=1.5, headwidth=7),
            ha='center', fontsize=10.5, fontweight='semibold',
            bbox=dict(boxstyle="round,pad=0.4", fc="#E8F8F5", ec=COLOR_KRIA, lw=1.2))

f1_path = os.path.join(output_dir, "hsi_1_fps_vs_platforms.png")
plt.savefig(f1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Guardada: {f1_path}")

# =========================================================================
# 2. FIGURA 2: Latencia de Inferencia por Parche y Tiempo Total por Escena
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5), dpi=300)

# Latencia en silicio por parche
b1 = ax1.bar(platforms, lat_patch_ms, color=colors, width=0.45, edgecolor="black", linewidth=1.2, zorder=3)
for b in b1:
    y = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2.0, y + 0.015, f"{y:.3f} ms", ha='center', va='bottom', fontweight='bold', color=COLOR_ACCENT)
ax1.set_ylabel("Latencia Media de Inferencia (ms)", fontweight='bold')
ax1.set_title("Latencia en Silicio Acelerador\n(Parche 13×13×30)", pad=12, fontweight='bold')
ax1.set_ylim(0, 0.75)
ax1.annotate(f"DPU es 41.5% más rápida\npor parche que Tensor Cores",
             xy=(0, 0.35), xytext=(0.5, 0.52),
             arrowprops=dict(facecolor=COLOR_KRIA, shrink=0.08, width=1.2, headwidth=6),
             ha='center', fontsize=9.5, fontweight='semibold',
             bbox=dict(boxstyle="round,pad=0.3", fc="#E8F8F5", ec=COLOR_KRIA, lw=1.0))

# Tiempo por escena completa (Inferencia vs Total Fin-a-Fin)
x = np.arange(len(platforms))
w = 0.32
b_infer = ax2.bar(x - w/2, infer_scene_s, w, label='Pura Inferencia (9,225 px)', color=[COLOR_KRIA, COLOR_JETSON], edgecolor="black", linewidth=1.1, zorder=3)
b_total = ax2.bar(x + w/2, total_scene_s, w, label='Total Fin-a-Fin (PCA + Extracción + Inf)', color=["#4DB6AC", "#A8D84E"], edgecolor="black", linewidth=1.1, zorder=3)

for b in b_infer:
    y = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2.0, y + 0.12, f"{y:.2f}s", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_ACCENT)
for b in b_total:
    y = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2.0, y + 0.12, f"{y:.2f}s", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color="#1B5E20")

ax2.set_ylabel("Tiempo por Escena Completa (segundos)", fontweight='bold')
ax2.set_title("Desglose Temporal por Escena\n(Indian Pines: 9,225 Píxeles)", pad=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(platforms, fontweight='semibold')
ax2.set_ylim(0, 8.0)
ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

f2_path = os.path.join(output_dir, "hsi_2_latency_vs_platforms.png")
plt.savefig(f2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Guardada: {f2_path}")

# =========================================================================
# 3. FIGURA 3: Consumo Energético y Potencia
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5), dpi=300)

# Potencia (W)
b1 = ax1.bar(platforms, power_w, color=colors, width=0.45, edgecolor="black", linewidth=1.2, zorder=3)
for b in b1:
    y = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2.0, y + 0.15, f"{y:.2f} W", ha='center', va='bottom', fontweight='bold', color=COLOR_ACCENT)
ax1.set_ylabel("Potencia Media Activa (Watts)", fontweight='bold')
ax1.set_title("Consumo de Potencia Activo", pad=12, fontweight='bold')
ax1.set_ylim(0, 8.5)

# Energía por Parche (mJ)
b2 = ax2.bar(platforms, energy_patch_mj, color=colors, width=0.45, edgecolor="black", linewidth=1.2, zorder=3)
for b in b2:
    y = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2.0, y + 0.08, f"{y:.3f} mJ", ha='center', va='bottom', fontweight='bold', color=COLOR_ACCENT)
ax2.set_ylabel("Energía por Parche de Inferencia (mJ)", fontweight='bold')
ax2.set_title("Eficiencia Energética por Inferencia en Silicio", pad=12, fontweight='bold')
ax2.set_ylim(0, 4.5)

# Efficiency annotation
energy_saving = (1.0 - (energy_patch_mj[0] / energy_patch_mj[1])) * 100.0
ax2.annotate(f"Kria KV260 ahorra {energy_saving:.1f}%\nde energía por parche",
             xy=(0, 2.2), xytext=(0.5, 3.2),
             arrowprops=dict(facecolor=COLOR_KRIA, shrink=0.08, width=1.3, headwidth=6),
             ha='center', fontsize=10.0, fontweight='semibold',
             bbox=dict(boxstyle="round,pad=0.35", fc="#E8F8F5", ec=COLOR_KRIA, lw=1.1))

f3_path = os.path.join(output_dir, "hsi_3_energy_vs_platforms.png")
plt.savefig(f3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Guardada: {f3_path}")

# =========================================================================
# 4. FIGURA 4: Precisión de Clasificación (OA, AA, Kappa)
# =========================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

x = np.arange(3)
width = 0.32

rects1 = ax.bar(x - width/2, [oa_values[0], aa_values[0], kappa_values[0]], width, 
                label='AMD Kria KV260 (DPU INT8)', color=COLOR_KRIA, edgecolor="black", linewidth=1.1, zorder=3)
rects2 = ax.bar(x + width/2, [oa_values[1], aa_values[1], kappa_values[1]], width, 
                label='NVIDIA Jetson Orin Nano (TRT INT8)', color=COLOR_JETSON, edgecolor="black", linewidth=1.1, zorder=3)

for rect in rects1:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.3, f"{h:.2f}%", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=COLOR_KRIA)

for rect in rects2:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.3, f"{h:.2f}%", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color="#4D7A00")

ax.set_ylabel("Exactitud (%)", fontweight='bold')
ax.set_title("Comparación de Precisión de Clasificación Hiperespectral\nModelo SS-ResNet INT8 en Dataset Indian Pines", pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(["Overall Accuracy (OA)", "Average Accuracy (AA)", "Kappa de Cohen (κ)"], fontweight='semibold')
ax.set_ylim(92, 102)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

f4_path = os.path.join(output_dir, "hsi_4_precision_vs_platforms.png")
plt.savefig(f4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Guardada: {f4_path}")

print("\n🎉 Todas las figuras HSI fueron generadas con éxito a 300 DPI con datos 100% empíricos.")
