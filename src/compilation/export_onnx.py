import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import onnx

def export_to_onnx(
    weights_path: str = "models/weights/yolo11n.pt",
    output_path: str = "models/onnx/yolo11n_640.onnx",
    img_size: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False
) -> str:
    """
    Exporta un modelo YOLO a formato ONNX optimizado.
    """
    output_path = os.path.abspath(output_path)
    weights_path = os.path.abspath(weights_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    model_name = Path(weights_path).stem
    if not os.path.exists(weights_path):
        print(f"⬇️  Pesos no encontrados localmente. Descargando {model_name}...")
        temp_model = YOLO(f"{model_name}.pt")
        # Si se descargó en cwd, moverlo a models/weights/
        if os.path.exists(f"{model_name}.pt"):
            shutil.move(f"{model_name}.pt", weights_path)
        print(f"✅ Pesos guardados en: {weights_path}")

    print(f"\n📦 [ONNX EXPORT] Exportando {model_name} a ONNX (imgsz={img_size}, opset={opset}, simplify={simplify})...")
    model = YOLO(weights_path)

    # Exportación con Ultralytics
    exported_file = model.export(
        format="onnx",
        imgsz=img_size,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic
    )

    if exported_file and os.path.exists(exported_file):
        if os.path.abspath(exported_file) != output_path:
            shutil.move(exported_file, output_path)
        print(f"✅ ONNX exportado exitosamente en: {output_path}")
    else:
        raise RuntimeError("Fallo en la exportación a ONNX.")

    # Validar grafo ONNX
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("🔍 Grafo ONNX verificado sin errores sintácticos.")

    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exportar YOLO a ONNX")
    parser.add_argument("--weights", type=str, default="models/weights/yolo11n.pt")
    parser.add_argument("--output", type=str, default="models/onnx/yolo11n_640.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        img_size=args.imgsz,
        opset=args.opset
    )
