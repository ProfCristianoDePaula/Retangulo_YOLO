from ultralytics import YOLO

# Carrega o modelo YOLO26n com pesos pré-treinados (transfer learning)
model = YOLO("yolo26n.pt")

# Treino com configuração de máxima qualidade para CPU
results = model.train(
    data="C:/Repositorios/Retangulo_YOLO/data.yaml",

    # --- Treino principal ---
    epochs=150,             # mais épocas = melhor convergência
    imgsz=640,              # resolução padrão YOLO — melhor qualidade de detecção
    batch=4,                # viável em RAM com 100 imagens
    device="cpu",
    workers=0,              # obrigatório no Windows
    amp=False,              # AMP (mixed precision) não funciona em CPU

    # --- Saídas ---
    project="C:/Repositorios/Retangulo_YOLO/runs/train",
    name="retangulo_v1",
    save_period=10,         # salva checkpoint a cada 10 epochs
    plots=True,             # gera gráficos de loss e mAP ao final
    verbose=True,

    # --- Convergência ---
    optimizer="SGD",        # SGD com momentum — melhor generalização
    momentum=0.937,
    weight_decay=0.0005,
    lr0=0.01,               # learning rate inicial
    lrf=0.01,               # fator final (lr_final = lr0 * lrf)
    cos_lr=True,            # scheduler cosseno — descida suave do LR
    warmup_epochs=5,        # aquecimento gradual nas primeiras épocas
    patience=25,            # early stopping — mais tolerância para CPU

    # --- Regularização ---
    label_smoothing=0.0,
    dropout=0.0,

    # --- Cache de imagens na RAM (100 imgs ~120MB, seguro) ---
    cache=True,

    # --- Augmentação (sem modificar — defaults são bem calibrados) ---
    mosaic=1.0,             # combina 4 imagens por amostra
    mixup=0.1,              # leve mistura entre pares de imagens
    degrees=10.0,           # rotação leve
    translate=0.1,
    scale=0.5,
    fliplr=0.5,             # espelhamento horizontal
    flipud=0.0,             # sem espelhamento vertical (retângulos têm orientação)
    hsv_h=0.015,            # variação de matiz
    hsv_s=0.7,              # variação de saturação
    hsv_v=0.4,              # variação de brilho

    val=True,
)

print("\nTreino concluído!")
print(f"Melhor modelo salvo em: {results.save_dir}/weights/best.pt")

# Validação final com o melhor modelo
best_model = YOLO(f"{results.save_dir}/weights/best.pt")
metrics = best_model.val(data="C:/Repositorios/Retangulo_YOLO/data.yaml")

print(f"\nmAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p[0]:.4f}")
print(f"Recall:    {metrics.box.r[0]:.4f}")