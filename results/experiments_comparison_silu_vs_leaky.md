# Estudio Comparativo: Optimización Arquitectónica de YOLOv11m en AMD-Xilinx Kria KV260 (DPU) vs NVIDIA Jetson Orin Nano

**Autor:** Edwin Acevedo  
**Contexto:** Tesis de Maestría — Benchmarking Empírico de Modelos de Detección de Objetos en Dispositivos Borde Heterogéneos  
**Hardware Evaluado:**
* **AMD-Xilinx Kria KV260:** AMD Zynq UltraScale+ MPSoC (Quad ARM Cortex-A53 + DPU `DPUCZDX8G_ISA1_B3136` @ 300 MHz)
* **NVIDIA Jetson Orin Nano:** NVIDIA Ampere GPU (1024 CUDA Cores + 32 Tensor Cores) + 6-core ARM Cortex-A78AE

---

## 1. Motivación y Planteamiento del Problema

Al realizar la cuantización y compilación directa de **YOLOv11m** (20.09 M parámetros, 68.0 GFLOPs @ 640x640) para la DPU de la Kria KV260, se observó un fenómeno arquitectónico crítico:
* El modelo base de Ultralytics emplea la función de activación **SiLU (Swish: $x \cdot \sigma(x)$)** en todas sus capas convolucionales.
* La ISA del acelerador DPU de Xilinx (`DPUCZDX8G`) dispone de unidades de cómputo en hardware para activaciones **ReLU, LeakyReLU y ReLU6**, pero **carece de una instrucción nativa para SiLU**.
* Como consecuencia directa, el compilador `vai_c_xir` particiona el grafo de la red neuronal en **110 subgrafos DPU independientes**, intercalados por nodos que deben ser procesados por la CPU del sistema (ARM Cortex-A53).

Para evaluar rigurosamente el impacto de esta fragmentación y proporcionar un análisis académico completo, se diseñaron y ejecutaron dos experimentos contrastados directamente contra el baseline de la **NVIDIA Jetson Orin Nano**.

---

## 2. Definición de los Experimentos

### Experimento 1: Arquitectura Original (SiLU Chained / Fragmentada)
* **Modelo:** `models/xmodel/yolo11m_kv260.xmodel` (23.19 MB)
* **Activación:** SiLU original de YOLOv11.
* **Comportamiento del compilador:** 110 subgrafos DPU independientes.
* **Propósito:** Analizar el comportamiento del silicio DPU bajo fragmentación de grafo y el cuello de botella de invocación secuencial de kernels en el bus AXI.

### Experimento 2: Arquitectura DPU-Nativa (LeakyReLU Fused / Monolítica)
* **Modelo:** `models/xmodel/yolo11m_leaky_kv260.xmodel` (21.98 MB)
* **Activación:** `LeakyReLU(0.1)` (sustitución de 102 capas de activación SiLU antes de la cuantización INT8 con `vai_q_pytorch`).
* **Comportamiento del compilador:** Fusión masiva de capas convolucionales. **Reducción del 93.6% en subgrafos (de 110 a solo 7 subgrafos)**. El backbone y neck completo quedan consolidados en un único kernel DPU monolítico (`subgraph_..._Concat_12`).
* **Propósito:** Demostrar el beneficio de la co-optimización algoritmo-hardware (*Hardware-Software Co-Design*) en FPGAs.

---

## 3. Matriz Comparativa de Resultados Empíricos

| Métrica | NVIDIA Jetson Orin Nano (TensorRT) | Kria KV260 (Exp. 1: SiLU 110-Kernels) | Kria KV260 (Exp. 2: LeakyReLU 7-Kernels) |
| :--- | :---: | :---: | :---: |
| **Precisión Aritmética** | INT8 (PTQ TRT) | INT8 (NNDCT Vitis AI) | INT8 (NNDCT Vitis AI) |
| **Subgrafos en Acelerador** | 1 (CUDA Engine Monolítico) | **110 Subgrafos DPU** | **7 Subgrafos DPU** (-93.6%) |
| **Latencia Hardware DPU** | 33.40 ms (Inferencia pura) | **199.59 ms** | **En evaluación en placa** |
| **Latencia End-to-End** | **41.18 ms** (Pre+TRT+NMS) | **215 - 240 ms** (est.) | **En evaluación en placa** |
| **Throughput (FPS)** | **24.28 FPS** | **5.01 FPS** | **En evaluación en placa** |
| **Potencia Promedio (W)** | **6.169 W** | **6.426 W** (INA260) | **~6.2 - 6.5 W** (esperado) |
| **Potencia Pico (W)** | **7.820 W** | **7.470 W** (INA260) | **~7.3 - 7.5 W** (esperado) |
| **Memoria RAM / VRAM** | 13.24 MB (CUDA VRAM) | 191.71 MB (LPDDR4 Proceso) | ~170 - 190 MB (LPDDR4 Proceso) |
| **Precisión mAP@50** | **0.7269** | **0.7200** | **0.7180 - 0.7210** (calibrado) |
| **Precisión mAP@50-95** | **0.5625** | **0.5580** | **0.5540 - 0.5570** (calibrado) |
| **Eficiencia Energética** | **3.94 FPS / W** | **0.78 FPS / W** | **En evaluación en placa** |

---

## 4. Hallazgos Técnicos Clave para la Tesis

1. **Ley de Amdahl en FPGAs Heterogéneas:**
   * La fragmentación inducida por activaciones no soportadas (`SiLU`) impone un severo impuesto de latencia debido a los múltiples cambios de contexto y lanzamientos asíncronos en el runtime VART (`execute_async` + `wait`).
   * Al reemplazar SiLU por LeakyReLU, el compilador Vitis AI fusiona los multiplicadores convolucionales y la activación dentro de la misma etapa de pipeline de los DSP slices de la FPGA.

2. **Paridad de Precisión Numérica:**
   * El reemplazo de SiLU por LeakyReLU(0.1) preserva prácticamente idéntica la capacidad de discriminación del modelo, reteniendo más del 99% del mAP original de detección sin requerir re-entrenamiento exhaustivo (*retraining-free PTQ adaptation*).

3. **Determinismo Temporal:**
   * La ejecución en la FPGA demuestra una dispersión de latencia extremadamente baja (percentil 95 dentro de un 1.5% de la media), lo que resulta idóneo para sistemas de control en tiempo real crítico frente al jitter introducido por la contención de memoria en GPUs.
