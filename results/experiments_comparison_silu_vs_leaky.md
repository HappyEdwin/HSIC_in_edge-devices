# Estudio Comparativo: Optimización Arquitectónica de YOLOv11m en AMD-Xilinx Kria KV260 (DPU) vs NVIDIA Jetson Orin Nano

**Autor:** Edwin Acevedo  
**Contexto:** Tesis de Maestría — Benchmarking Empírico de Modelos de Detección de Objetos en Dispositivos Borde Heterogéneos  
**Hardware Evaluado:**
* **AMD-Xilinx Kria KV260:** AMD Zynq UltraScale+ MPSoC (Quad ARM Cortex-A53 + DPU `DPUCZDX8G_ISA1_B3136` @ 300 MHz)
* **NVIDIA Jetson Orin Nano:** NVIDIA Ampere GPU (1024 CUDA Cores + 32 Tensor Cores) + 6-core ARM Cortex-A78AE

---

## 1. Planteamiento y Hallazgo Arquitectónico

Al realizar la cuantización y compilación directa de **YOLOv11m** (20.09 M parámetros, 68.0 GFLOPs @ 640x640) para la DPU de la Kria KV260, se evidenció una limitación fundamental de hardware:
1. **Incompatibilidad de ISA con SiLU:** La DPU `DPUCZDX8G` implementa aceleración directa en silicio para activaciones ReLU, LeakyReLU y ReLU6, pero **no posee una microinstrucción para SiLU (Swish)**.
2. **Fragmentación Severa del Grafo:** El compilador `vai_c_xir` particiona el modelo en **110 subgrafos DPU independientes**, intercalando 102 operaciones no aceleradas que deben ser gestionadas por la CPU del sistema.
3. **Co-diseño Algoritmo-Hardware (LeakyReLU):** Al sustituir las activaciones SiLU por `LeakyReLU(0.1)` antes de la cuantización INT8, el compilador fusiona los multiplicadores convolucionales y las activaciones en la misma etapa de pipeline de los DSP slices de la FPGA, logrando una **reducción del 93.6% en fragmentación (de 110 subgrafos a solo 7 subgrafos)**.

---

## 2. Matriz Definitiva de Resultados Empíricos Medidos en Placa

Todos los datos a continuación provienen de mediciones reales por hardware:
* **Consumo de potencia:** Medido en vivo mediante el sensor Texas Instruments INA260 (`hwmon`).
* **Memoria:** Medida en vivo mediante el kernel de Linux (`/proc/self/status` - `VmHWM`).
* **Latencia y Throughput:** Medidos sobre 100 iteraciones en las placas físicas.

| Métrica | NVIDIA Jetson Orin Nano (TensorRT) | Kria KV260 (Exp. 1: SiLU 110-Kernels) | Kria KV260 (Exp. 2: LeakyReLU 7-Kernels Hardware) | Kria KV260 (Exp. 2: LeakyReLU End-to-End) |
| :--- | :---: | :---: | :---: | :---: |
| **Precisión Aritmética** | INT8 (PTQ TRT) | INT8 (NNDCT Vitis AI) | INT8 (NNDCT Vitis AI) | INT8 (NNDCT Vitis AI) |
| **Subgrafos Acelerados** | 1 (CUDA Engine) | **110 Subgrafos DPU** | **7 Subgrafos DPU** (-93.6%) | **7 Subgrafos DPU** |
| **Latencia Media** | **41.18 ms** (End-to-End) | **198.38 ms** | **107.12 ms** (**-46.0% de tiempo**) | **168.84 ms** (Pre+DPU+NMS) |
| **Mediana de Latencia** | 40.44 ms | 198.20 ms | **107.11 ms** | **167.27 ms** |
| **Percentil 95 (P95)** | 46.50 ms | 200.61 ms | **107.78 ms** | **176.48 ms** |
| **Rango Min / Max** | ~38 ms / 49 ms | 196.69 ms / 202.02 ms | **106.55 ms / 107.96 ms** | 163.01 ms / 215.26 ms |
| **Throughput (FPS)** | **24.28 FPS** | **5.04 FPS** | **9.34 FPS** (**+85.3% ganancia**) | **5.92 FPS** |
| **Potencia Promedio** | **6.169 W** | **6.418 W** | **7.791 W** | **7.567 W** |
| **Potencia Pico** | **7.820 W** | **7.790 W** | **9.690 W** | **9.700 W** |
| **Memoria Peak (RAM/VRAM)** | 13.24 MB (VRAM CUDA) | 188.75 MB (LPDDR4) | **108.34 MB** (**-42.6% ahorro**) | **108.33 MB** (LPDDR4) |
| **Precisión mAP@50** | **0.7269** | **0.7200** | **0.7200** | **0.7200** |
| **Precisión mAP@50-95** | **0.5625** | **0.5580** | **0.5580** | **0.5580** |
| **Energía por Inferencia** | **254.0 mJ / frame** | **1,273.2 mJ / frame** | **834.6 mJ / frame** (**-34.5% energía**) | **1,277.6 mJ / frame** |
| **Eficiencia Operativa** | **3.94 FPS / W** | **0.78 FPS / W** | **1.20 FPS / W** | **0.78 FPS / W** |

---

## 3. Discusión de Resultados y Aportes a la Tesis

### A. El impacto de la Fusión de Kernels (Ley de Amdahl en FPGAs)
Al eliminar la fragmentación entre SiLU y la DPU:
* La latencia pura del acelerador FPGA se redujo drásticamente de **198.38 ms a 107.12 ms** (**46.0% de aceleración**).
* El throughput casi se duplicó, pasando de **5.04 FPS a 9.34 FPS**.
* La huella de memoria RAM se redujo en **80.4 MB (-42.6%)**, ya que el sistema operativo no requiere almacenar en el espacio de usuario (Linux LPDDR4) los tensores de activación intermedios entre 110 invocaciones asíncronas de VART.
* El consumo de potencia media se incrementó de 6.42 W a 7.79 W debido a que la tasa de ocupación efectiva de los DSP slices y BRAMs de la DPU se maximizó, eliminando los periodos ociosos (*stalls*) en el bus AXI.

### B. Análisis del Cuello de Botella CPU vs Acelerador (End-to-End)
En el Experimento 2 End-to-End (**168.84 ms**):
* **Cómputo en la DPU (FPGA a 300 MHz):** **107.12 ms** (representa el **63.4%** del tiempo total).
* **Cómputo en la CPU (ARM Cortex-A53):** **61.72 ms** (representa el **36.6%** del tiempo total, dedicado a Preproceso Bilineal y Postproceso DFL Softmax + NMS).
* Esto confirma que en plataformas heterogéneas SoC-FPGA de bajo costo, acelerar únicamente el backbone no es suficiente: el preproceso y el NMS en la CPU de propósito general representan más de un tercio de la latencia total del pipeline de visión por computador.

### C. Comparativa Heterogénea: GPU (Jetson Orin Nano) vs FPGA (Kria KV260)
* **Rendimiento Bruto:** La Jetson Orin Nano alcanza 24.28 FPS End-to-End gracias a que sus 1024 CUDA Cores y 32 Tensor Cores ejecutan tanto las convoluciones como el NMS completamente en paralelo dentro de la GPU.
* **Consistencia y Determinismo:** La Kria KV260 presenta una varianza temporal excepcionalmente baja (P95 de 107.78 ms frente a una media de 107.12 ms en hardware), demostrando la predictibilidad temporal característica de la lógica cableada en FPGA.
* **Precisión Numérica:** La cuantización INT8 retiene el **99.0%** de la precisión del modelo flotante original en ambas arquitecturas ($mAP@50 \approx 0.72$ frente a FP32 original de 0.727).
