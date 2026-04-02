import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = "C:/Repositorios/Retangulo_YOLO/runs/train/retangulo_v1/weights/best.pt"

model = YOLO(MODEL_PATH)


def selecionar_imagem():
    caminho = filedialog.askopenfilename(
        title="Selecionar imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos os arquivos", "*.*")]
    )
    if not caminho:
        return

    label_caminho.config(text=caminho)

    # Inferência
    conf = slider_conf.get() / 100
    results = model.predict(source=caminho, conf=conf, verbose=False)
    result = results[0]

    # Atualiza métricas
    boxes = result.boxes
    n_deteccoes = len(boxes)
    label_deteccoes.config(text=f"Detecções: {n_deteccoes}")

    if n_deteccoes > 0:
        confs = [f"{c:.2f}" for c in boxes.conf.tolist()]
        label_conf_val.config(text=f"Confiança(s): {', '.join(confs)}")
    else:
        label_conf_val.config(text="Confiança(s): —")

    # Renderiza imagem com bounding boxes
    img_annotada = result.plot()                        # BGR numpy array
    img_rgb = cv2.cvtColor(img_annotada, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Redimensiona para caber no canvas mantendo proporção
    max_w, max_h = 700, 500
    img_pil.thumbnail((max_w, max_h), Image.LANCZOS)

    foto = ImageTk.PhotoImage(img_pil)
    canvas.config(width=img_pil.width, height=img_pil.height)
    canvas.image = foto                                  # evita garbage collection
    canvas.create_image(0, 0, anchor="nw", image=foto)


# ── Janela principal ─────────────────────────────────────────
root = tk.Tk()
root.title("YOLO26n — Detecção de Retângulos")
root.resizable(False, False)

# ── Painel de controle (topo) ─────────────────────────────────
frame_top = tk.Frame(root, pady=8, padx=10)
frame_top.pack(fill="x")

btn_abrir = tk.Button(frame_top, text="Abrir Imagem", command=selecionar_imagem,
                      font=("Segoe UI", 10, "bold"), bg="#0078D4", fg="white",
                      padx=12, pady=4, relief="flat", cursor="hand2")
btn_abrir.pack(side="left", padx=(0, 12))

tk.Label(frame_top, text="Confiança mínima:", font=("Segoe UI", 9)).pack(side="left")
slider_conf = tk.Scale(frame_top, from_=10, to=95, orient="horizontal",
                       length=160, font=("Segoe UI", 9))
slider_conf.set(50)
slider_conf.pack(side="left", padx=(4, 0))

# ── Caminho da imagem ─────────────────────────────────────────
label_caminho = tk.Label(root, text="Nenhuma imagem selecionada",
                         font=("Segoe UI", 8), fg="#555", anchor="w", padx=10)
label_caminho.pack(fill="x")

# ── Canvas da imagem ──────────────────────────────────────────
canvas = tk.Canvas(root, bg="#1e1e1e", width=700, height=460)
canvas.pack(padx=10, pady=(4, 0))

# ── Métricas (rodapé) ─────────────────────────────────────────
frame_bottom = tk.Frame(root, pady=6, padx=10)
frame_bottom.pack(fill="x")

label_deteccoes = tk.Label(frame_bottom, text="Detecções: —",
                            font=("Segoe UI", 10, "bold"), fg="#0078D4")
label_deteccoes.pack(side="left", padx=(0, 20))

label_conf_val = tk.Label(frame_bottom, text="Confiança(s): —",
                           font=("Segoe UI", 10))
label_conf_val.pack(side="left")

root.mainloop()
