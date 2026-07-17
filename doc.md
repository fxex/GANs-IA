# Documentación GAN - Fashion MNIST

## Arquitectura del Generador

```mermaid
flowchart TD
    A["Entrada: Ruido Z<br/>(batch, 100)"] --> B["Dense(128)<br/>he_uniform"]
    B --> C["LeakyReLU(α=0.2)"]
    C --> D["Dense(256)<br/>he_uniform"]
    D --> E["BatchNorm<br/>(momentum=0.1)"]
    E --> F["LeakyReLU(α=0.2)"]
    F --> G["Dense(512)<br/>he_uniform"]
    G --> H["BatchNorm<br/>(momentum=0.1)"]
    H --> I["LeakyReLU(α=0.2)"]
    I --> J["Dense(1024)<br/>he_uniform"]
    J --> K["BatchNorm<br/>(momentum=0.1)"]
    K --> L["LeakyReLU(α=0.2)"]
    L --> M["Dense(784, tanh)<br/>he_uniform"]
    M --> N["Reshape(28, 28, 1)"]
    N --> O["Salida: Imagen Generada<br/>(batch, 28, 28, 1)"]

    style A fill:#3b82f6,color:#fff
    style O fill:#10b981,color:#fff
    style B fill:#f59e0b,color:#000
    style D fill:#f59e0b,color:#000
    style G fill:#f59e0b,color:#000
    style J fill:#f59e0b,color:#000
    style M fill:#f59e0b,color:#000
    style E fill:#a855f7,color:#fff
    style H fill:#a855f7,color:#fff
    style K fill:#a855f7,color:#fff
    style C fill:#ef4444,color:#fff
    style F fill:#ef4444,color:#fff
    style I fill:#ef4444,color:#fff
    style L fill:#ef4444,color:#fff
    style N fill:#6b7280,color:#fff
```

**Resumen de capas:**

| # | Capa | Forma de salida |
|---|------|-----------------|
| 1 | Dense(128) + he_uniform | (batch, 128) |
| 2 | LeakyReLU(0.2) | (batch, 128) |
| 3 | Dense(256) + he_uniform | (batch, 256) |
| 4 | BatchNormalization | (batch, 256) |
| 5 | LeakyReLU(0.2) | (batch, 256) |
| 6 | Dense(512) + he_uniform | (batch, 512) |
| 7 | BatchNormalization | (batch, 512) |
| 8 | LeakyReLU(0.2) | (batch, 512) |
| 9 | Dense(1024) + he_uniform | (batch, 1024) |
| 10 | BatchNormalization | (batch, 1024) |
| 11 | LeakyReLU(0.2) | (batch, 1024) |
| 12 | Dense(784, tanh) + he_uniform | (batch, 784) |
| 13 | Reshape(28, 28, 1) | (batch, 28, 28, 1) |

---

## Arquitectura del Discriminador (Antagónico)

```mermaid
flowchart TD
    A["Entrada: Imagen<br/>(batch, 28, 28, 1)"] --> B["Flatten"]
    B --> C["Dense(512)<br/>he_uniform"]
    C --> D["LeakyReLU(α=0.2)"]
    D --> E["Dense(256)<br/>he_uniform"]
    E --> F["LeakyReLU(α=0.2)"]
    F --> G["Dense(1, sigmoid)<br/>he_uniform"]
    G --> H["Salida: Probabilidad<br/>(batch, 1)<br/>0=Falso, 1=Real"]

    style A fill:#3b82f6,color:#fff
    style H fill:#10b981,color:#fff
    style C fill:#f59e0b,color:#000
    style E fill:#f59e0b,color:#000
    style G fill:#f59e0b,color:#000
    style D fill:#ef4444,color:#fff
    style F fill:#ef4444,color:#fff
    style B fill:#6b7280,color:#fff
```

**Resumen de capas:**

| # | Capa | Forma de salida |
|---|------|-----------------|
| 1 | Flatten | (batch, 784) |
| 2 | Dense(512) + he_uniform | (batch, 512) |
| 3 | LeakyReLU(0.2) | (batch, 512) |
| 4 | Dense(256) + he_uniform | (batch, 256) |
| 5 | LeakyReLU(0.2) | (batch, 256) |
| 6 | Dense(1, sigmoid) + he_uniform | (batch, 1) |

---

## Flujo de Entrenamiento

```mermaid
flowchart TD
    subgraph Datos["Pipeline de Datos"]
        D1["Fashion-MNIST<br/>(60,000 imágenes)"] --> D2["Reshape → (N, 28, 28, 1)"]
        D2 --> D3["Normalizar: (pixel - 127.5) / 127.5<br/>(rango [-1, +1])"]
        D3 --> D4["tf.data.Dataset<br/>shuffle(60000) → batch(128)"]
    end

    subgraph Optim["Optimizadores"]
        O1["Adam<br/>lr=0.0002<br/>β₁=0.5, β₂=0.999"]
        O2["Adam<br/>lr=0.0002<br/>β₁=0.5, β₂=0.999"]
    end

    subgraph Step["train_step (por cada batch)"]
        S1["Muestrear ruido Z<br/>~ N(0,1) shape=(128, 100)"] --> S2["G(Z) → Imágenes Falsas"]
        S2 --> S3{"GradientTape"}
        S3 --> S4["D(imágenes_reales) → real_output"]
        S3 --> S5["D(imágenes_falsas) → fake_output"]
        S4 --> S6["loss_gen = BCE(1, fake_output)"]
        S5 --> S6
        S4 --> S7["loss_disc = BCE(1, real_output) + BCE(0, fake_output)"]
        S5 --> S7
        S6 --> S8["Gradientes: ∂loss_gen / ∂θ_G"]
        S7 --> S9["Gradientes: ∂loss_disc / ∂θ_D"]
        S8 --> S10["O_G.apply_gradients()<br/>Actualiza pesos del Generador"]
        S9 --> S11["O_D.apply_gradients()<br/>Actualiza pesos del Discriminador"]
    end

    subgraph Loop["Bucle de Entrenamiento"]
        L1["Época 1..100"] --> L2["Por cada batch de imágenes"]
        L2 --> L3["Ejecutar train_step"]
        L3 --> L4["Calcular media y std<br/>de loss_gen y loss_disc"]
        L4 --> L5["Guardar 25 imágenes fijas<br/>→ training_images/epoch_NNNN.png"]
        L5 --> L2
    end

    subgraph Loss["Funciones de Pérdida"]
        LS["BinaryCrossentropy<br/>(sigmoid)"]
    end

    D4 --> L1
    O1 --> S10
    O2 --> S11
    LS -.-> S6
    LS -.-> S7

    L4 --> L6["Al finalizar:<br/>Gráfica de curvas de loss<br/>(escala logarítmica)"]

    style D1 fill:#3b82f6,color:#fff
    style S1 fill:#60a5fa,color:#fff
    style S2 fill:#f59e0b,color:#000
    style S4 fill:#a855f7,color:#fff
    style S5 fill:#a855f7,color:#fff
    style S6 fill:#ef4444,color:#fff
    style S7 fill:#ef4444,color:#fff
    style S8 fill:#10b981,color:#fff
    style S9 fill:#10b981,color:#fff
    style S10 fill:#10b981,color:#fff
    style S11 fill:#10b981,color:#fff
    style LS fill:#6b7280,color:#fff
```

**Resumen del entrenamiento:**

| Componente | Detalle |
|------------|---------|
| Pérdida | BinaryCrossentropy (sigmoid) |
| Pérdida del Generador | `BCE(label=1, D(G(Z)))` — engañar al Discriminador |
| Pérdida del Discriminador | `BCE(1, D(real)) + BCE(0, D(fake))` — clasificar correctamente |
| Optimizador | Adam (lr=0.0002, β₁=0.5, β₂=0.999) — uno por modelo |
| Ratio G:D | 1 : 1 (un paso de gradiente por modelo por batch) |
| Batch size | 128 |
| Épocas | 100 |
| Fijación de ruido | Semilla=42, 25 vectores fijos para visualizar progreso |
