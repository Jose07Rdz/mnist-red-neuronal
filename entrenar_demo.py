"""Entrena los dos modelos de la demo web y exporta sus pesos a docs/modelos.json.

Son versiones pequeñas del MLP y la CNN del notebook, para que el navegador pueda
calcular y dibujar cada capa. Se entrenan con aumento de datos (rotación, escala y
desplazamiento) porque los dígitos dibujados con el mouse no se ven como los de MNIST.

Uso:
    uv run python entrenar_demo.py
"""

import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
SALIDA = Path("docs/modelos.json")
EPOCAS = 8

normalizar = transforms.Normalize((0.1307,), (0.3081,))
aumento = transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    normalizar,
])
simple = transforms.Compose([transforms.ToTensor(), normalizar])

entrenamiento = datasets.MNIST("data", train=True, download=True, transform=aumento)
prueba = datasets.MNIST("data", train=False, download=True, transform=simple)
cargador_entrenamiento = DataLoader(entrenamiento, batch_size=128, shuffle=True)
cargador_prueba = DataLoader(prueba, batch_size=1000)


def crear_mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 10),
    )


def crear_cnn() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # [8, 14, 14]
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [16, 7, 7]
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 10),
    )


def entrenar(modelo: nn.Module, nombre: str) -> float:
    modelo.to(device)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    programa = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, EPOCAS)
    perdida_fn = nn.CrossEntropyLoss()
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        for imagenes, etiquetas in cargador_entrenamiento:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            perdida = perdida_fn(modelo(imagenes), etiquetas)
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
        programa.step()
        precision = evaluar(modelo)
        print(f"{nombre} época {epoca}: precisión en prueba {precision:.2%}")
    return precision


@torch.no_grad()
def evaluar(modelo: nn.Module) -> float:
    modelo.eval()
    aciertos = 0
    for imagenes, etiquetas in cargador_prueba:
        aciertos += (modelo(imagenes.to(device)).argmax(1).cpu() == etiquetas).sum().item()
    return aciertos / len(prueba)


def a_base64(t: torch.Tensor) -> str:
    """Float32 little-endian en base64: más compacto que una lista de números en JSON."""
    return base64.b64encode(t.detach().cpu().float().numpy().astype("<f4").tobytes()).decode()


def exportar_capas(modelo: nn.Sequential) -> list[dict]:
    capas = []
    for m in modelo:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            capas.append({
                "tipo": "linear" if isinstance(m, nn.Linear) else "conv",
                "forma": list(m.weight.shape),
                "pesos": a_base64(m.weight),
                "sesgo": a_base64(m.bias),
            })
    return capas


def main() -> None:
    mlp, cnn = crear_mlp(), crear_cnn()
    precision_mlp = entrenar(mlp, "MLP")
    precision_cnn = entrenar(cnn, "CNN")

    # Casos de prueba para comprobar que el cálculo en JavaScript da lo mismo que PyTorch.
    imagenes = torch.stack([prueba[i][0] for i in range(5)])
    mlp.eval(), cnn.eval()
    with torch.no_grad():
        casos = {
            "imagenes": a_base64(imagenes),
            "logits_mlp": a_base64(mlp(imagenes.to(device))),
            "logits_cnn": a_base64(cnn(imagenes.to(device))),
        }

    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(json.dumps({
        "normalizacion": {"media": 0.1307, "desviacion": 0.3081},
        "mlp": {"precision": round(precision_mlp, 4), "capas": exportar_capas(mlp)},
        "cnn": {"precision": round(precision_cnn, 4), "capas": exportar_capas(cnn)},
    }))
    Path("data/casos_prueba.json").write_text(json.dumps(casos))
    print(f"Guardado {SALIDA} ({SALIDA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
