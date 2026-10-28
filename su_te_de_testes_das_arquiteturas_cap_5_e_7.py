"""
Suíte Prática de Validação de Arquiteturas - Deep Learning for Vision Systems (Mohamed Elgendy)
Capítulos 5 (Advanced CNN Architectures) e 7 (Object Detection: R-CNN, SSD, YOLO)

Este script corrige todos os erros de tradução/OCR presentes no PDF em português
(ex: 'Sequencial()' -> 'Sequential()', 'Adicionar()' -> 'Add()', 'eixo=3' -> 'axis=3')
e executa testes reais de construção de grafos, passagem direta (forward pass),
contagem de parâmetros e algoritmos geométricos de detecção de objetos.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Activation,
    BatchNormalization,
    Dropout,
    Add,
    Concatenate,
    GlobalAveragePooling2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


# =====================================================================
# PARTE 1: ARQUITETURAS DE CLASSIFICAÇÃO (CAPÍTULO 5)
# =====================================================================

def build_lenet5(input_shape=(32, 32, 1), num_classes=10) -> Model:
    """
    Seção 5.2.2: Implementação exata da LeNet-5 (1998).
    Nota técnica: Na página 220 do PDF, o código mostra padding='same' na C1, mas com entrada
    32x32x1 e kernel 5x5, para produzir (None, 28, 28, 6) e 61.706 parâmetros (Figura 5.5),
    o padding correto da camada C1 é 'valid'.
    """
    model = Sequential(name="LeNet-5")
    # C1: Camada Convolucional (32x32x1 -> 28x28x6)
    model.add(
        Conv2D(
            filters=6,
            kernel_size=(5, 5),
            strides=1,
            activation="tanh",
            input_shape=input_shape,
            padding="valid",
            name="C1_conv",
        )
    )
    # S2: Subamostragem Average Pooling (28x28x6 -> 14x14x6)
    model.add(
        AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid", name="S2_avg_pool")
    )
    # C3: Camada Convolucional (14x14x6 -> 10x10x16)
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=1,
            activation="tanh",
            padding="valid",
            name="C3_conv",
        )
    )
    # S4: Subamostragem Average Pooling (10x10x16 -> 5x5x16)
    model.add(
        AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid", name="S4_avg_pool")
    )
    # C5: Camada Convolucional (5x5x16 -> 1x1x120)
    model.add(
        Conv2D(
            filters=120,
            kernel_size=(5, 5),
            strides=1,
            activation="tanh",
            padding="valid",
            name="C5_conv",
        )
    )
    model.add(Flatten(name="flatten"))
    # FC6: Camada Totalmente Conectada (120 -> 84)
    model.add(Dense(units=84, activation="tanh", name="FC6_dense"))
    # FC7: Saída Softmax (84 -> 10)
    model.add(Dense(units=num_classes, activation="softmax", name="FC7_output"))
    return model


def lenet_lr_schedule(epoch: int) -> float:
    """Seção 5.2.3: Agendamento de taxa de aprendizado da LeNet-5."""
    lr = 5e-4
    if epoch > 9:
        lr = 1e-5
    elif epoch > 5:
        lr = 5e-5
    elif epoch > 2:
        lr = 2e-4
    return lr


def build_alexnet(input_shape=(227, 227, 3), num_classes=1000) -> Model:
    """
    Seção 5.3.3: Implementação da AlexNet (2012) com BatchNormalization e L2(0.0005).
    Usa 227x227x3 conforme explicado na página 226 para gerar exatamente 55x55 na CONV1
    e 6x6x256 (9216 features) no Flatten, resultando em ~62,38 milhões de parâmetros.
    """
    weight_decay = 0.0005
    model = Sequential(name="AlexNet")

    # 1ª Camada (CONV1 + MaxPool + BatchNorm) -> 227x227x3 para 55x55x96 -> Pool 27x27x96
    model.add(
        Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding="valid",
            input_shape=input_shape,
            kernel_regularizer=l2(weight_decay),
        )
    )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 2ª Camada (CONV2 + MaxPool + BatchNorm) -> 27x27x256 -> Pool 13x13x256
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(weight_decay),
        )
    )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 3ª Camada (CONV3 + BatchNorm, sem Pool) -> 13x13x384
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(weight_decay),
        )
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # 4ª Camada (CONV4 + BatchNorm, sem Pool) -> 13x13x384
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(weight_decay),
        )
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # 5ª Camada (CONV5 + BatchNorm + MaxPool) -> 13x13x256 -> Pool 6x6x256
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(weight_decay),
        )
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Achatar (6x6x256 = 9216) + Classificador Denso (FC6, FC7, FC8)
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))

    return model


def build_vgg16(input_shape=(224, 224, 3), num_classes=1000) -> Model:
    """
    Seção 5.4.3: Implementação da Configuração D (VGG-16).
    13 camadas CONV (todas 3x3, stride 1, padding 'same') + 3 camadas FC.
    Total exato esperado: 138.357.544 parâmetros.
    """
    model = Sequential(name="VGG-16")

    # Bloco #1 (2x CONV 64 + Pool)
    model.add(
        Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same", input_shape=input_shape)
    )
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Bloco #2 (2x CONV 128 + Pool)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Bloco #3 (3x CONV 256 + Pool)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Bloco #4 (3x CONV 512 + Pool)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Bloco #5 (3x CONV 512 + Pool)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Bloco #6 (Classificador Denso)
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def inception_module(
    x,
    filters_1x1: int,
    filters_3x3_reduce: int,
    filters_3x3: int,
    filters_5x5_reduce: int,
    filters_5x5: int,
    filters_pool_proj: int,
    name: str = None,
):
    """
    Seção 5.5.5: Módulo Inception com Redução de Dimensionalidade (Camadas de Gargalo 1x1).
    Processa o tensor em 4 rotas paralelas e concatena ao longo do eixo de canais (axis=-1).
    """
    # Rota 1: Convolução 1x1 direta
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding="same", activation="relu")(x)

    # Rota 2: Gargalo 1x1 -> Convolução 3x3
    pre_conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding="same", activation="relu")(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding="same", activation="relu")(pre_conv_3x3)

    # Rota 3: Gargalo 1x1 -> Convolução 5x5
    pre_conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding="same", activation="relu")(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding="same", activation="relu")(pre_conv_5x5)

    # Rota 4: MaxPool 3x3 (stride 1) -> Projeção 1x1
    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding="same", activation="relu")(pool_proj)

    # Concatenação na profundidade (eixo dos canais = 3 ou -1)
    output = Concatenate(axis=-1, name=name)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    return output


def build_googlenet(input_shape=(224, 224, 3), num_classes=1000) -> Model:
    """
    Seção 5.5.5: Arquitetura GoogLeNet (Inception v1) completa seguindo a Tabela da Figura 5.24.
    Dividida nas Partes A (Stem), B (9 módulos Inception: 3a-3b, 4a-4e, 5a-5b) e C (Classificador).
    """
    input_layer = Input(shape=input_shape)

    # Parte A: Camadas iniciais (Stem)
    x = Conv2D(64, (7, 7), padding="same", strides=(2, 2), activation="relu", name="conv1_7x7_s2")(
        input_layer
    )
    x = MaxPooling2D((3, 3), padding="same", strides=(2, 2), name="maxpool1_3x3_s2")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 1), padding="same", strides=(1, 1), activation="relu", name="conv2_1x1")(x)
    x = Conv2D(192, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv2_3x3")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding="same", strides=(2, 2), name="maxpool2_3x3_s2")(x)

    # Parte B: 9 Módulos Inception empilhados
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name="inception_3a")
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name="inception_3b")
    x = MaxPooling2D((3, 3), padding="same", strides=(2, 2), name="maxpool3_3x3_s2")(x)

    x = inception_module(x, 192, 96, 208, 16, 48, 64, name="inception_4a")
    x = inception_module(x, 160, 112, 224, 24, 64, 64, name="inception_4b")
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name="inception_4c")
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name="inception_4d")
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name="inception_4e")
    x = MaxPooling2D((3, 3), padding="same", strides=(2, 2), name="maxpool4_3x3_s2")(x)

    x = inception_module(x, 256, 160, 320, 32, 128, 128, name="inception_5a")
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name="inception_5b")

    # Parte C: Classificador com AveragePooling 7x7 + Dropout(0.4)
    x = AveragePooling2D(pool_size=(7, 7), strides=1, padding="valid", name="avgpool_7x7")(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=input_layer, outputs=output, name="GoogLeNet")


def bottleneck_residual_block(X, kernel_size: int, filters: list, reduce: bool = False, s: int = 2):
    """
    Seção 5.6.2: Bloco Residual com Gargalo (1x1 -> 3x3 -> 1x1) da ResNet-50.
    Suporta atalho regular (identidade) e atalho de redução (projeção 1x1 com stride s).
    """
    F1, F2, F3 = filters
    X_shortcut = X

    if reduce:
        # Atalho de redução para igualar canais (F3) e dimensão espacial (stride s)
        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid")(
            X_shortcut
        )
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

        # Primeiro componente do caminho principal com stride s
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)
    else:
        # Primeiro componente do caminho principal com stride 1
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

    # Segundo componente do caminho principal (3x3 CONV com padding 'same')
    X = Conv2D(filters=F2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    # Terceiro componente do caminho principal (1x1 CONV restaura dimensão F3)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid")(X)
    X = BatchNormalization(axis=3)(X)

    # Etapa final: Soma caminho principal f(x) + atalho x antes da ativação ReLU
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def build_resnet50(input_shape=(224, 224, 3), classes=1000) -> Model:
    """
    Seção 5.6.3: Construção da ResNet-50 completa (50 camadas de peso:
    1 conv inicial + [3 + 4 + 6 + 3]*3 convs nos blocos residuais + 1 dense final = 50).
    """
    X_input = Input(input_shape)

    # Etapa 1
    X = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1")(X_input)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(X)

    # Etapa 2 (3 blocos residuais = 9 camadas CONV)
    X = bottleneck_residual_block(X, 3, [64, 64, 256], reduce=True, s=1)
    X = bottleneck_residual_block(X, 3, [64, 64, 256])
    X = bottleneck_residual_block(X, 3, [64, 64, 256])

    # Etapa 3 (4 blocos residuais = 12 camadas CONV)
    X = bottleneck_residual_block(X, 3, [128, 128, 512], reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [128, 128, 512])
    X = bottleneck_residual_block(X, 3, [128, 128, 512])
    X = bottleneck_residual_block(X, 3, [128, 128, 512])

    # Etapa 4 (6 blocos residuais = 18 camadas CONV)
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], reduce=True, s=2)
    for _ in range(5):
        X = bottleneck_residual_block(X, 3, [256, 256, 1024])

    # Etapa 5 (3 blocos residuais = 9 camadas CONV)
    X = bottleneck_residual_block(X, 3, [512, 512, 2048], reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [512, 512, 2048])
    X = bottleneck_residual_block(X, 3, [512, 512, 2048])

    # Global Average Pooling + Classificador Softmax
    X = GlobalAveragePooling2D(name="avg_pool")(X)
    X = Dense(classes, activation="softmax", name=f"fc{classes}")(X)

    return Model(inputs=X_input, outputs=X, name="ResNet50")


# =====================================================================
# PARTE 2: ALGORITMOS E COMPONENTES DE DETECÇÃO DE OBJETOS (CAPÍTULO 7)
# =====================================================================

def compute_iou(box_a: list, box_b: list) -> float:
    """
    Seção 7.1.4: Calcula a Interseção sobre União (IoU) entre duas caixas no formato
    centro + largura/altura: [x_center, y_center, width, height].
    """
    # Converter de [x_c, y_c, w, h] para [x_min, y_min, x_max, y_max]
    a_xmin, a_ymin = box_a[0] - box_a[2] / 2.0, box_a[1] - box_a[3] / 2.0
    a_xmax, a_ymax = box_a[0] + box_a[2] / 2.0, box_a[1] + box_a[3] / 2.0

    b_xmin, b_ymin = box_b[0] - box_b[2] / 2.0, box_b[1] - box_b[3] / 2.0
    b_xmax, b_ymax = box_b[0] + box_b[2] / 2.0, box_b[1] + box_b[3] / 2.0

    # Coordenadas do retângulo de interseção
    inter_xmin = max(a_xmin, b_xmin)
    inter_ymin = max(a_ymin, b_ymin)
    inter_xmax = min(a_xmax, b_xmax)
    inter_ymax = min(a_ymax, b_ymax)

    inter_width = max(0.0, inter_xmax - inter_xmin)
    inter_height = max(0.0, inter_ymax - inter_ymin)
    area_overlap = inter_width * inter_height

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    area_union = area_a + area_b - area_overlap

    return area_overlap / area_union if area_union > 0 else 0.0


def non_maximum_suppression(
    boxes: list, scores: list, conf_threshold: float = 0.5, iou_threshold: float = 0.45
) -> list:
    """
    Seção 7.1.3 & 7.3.4: Implementação do algoritmo Non-Maximum Suppression (NMS).
    Retorna os índices das caixas selecionadas após filtrar por confiança e suprimir sobreposições.
    """
    # 1. Descartar caixas abaixo do limiar de confiança
    valid_indices = [i for i, s in enumerate(scores) if s >= conf_threshold]
    # 2. Ordenar em ordem decrescente de pontuação de objetividade/confiança
    sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)

    selected_indices = []
    while sorted_indices:
        # Selecionar a caixa com maior probabilidade atual
        current = sorted_indices.pop(0)
        selected_indices.append(current)

        # 3 & 4. Suprimir caixas restantes cujo IoU com a caixa selecionada supere iou_threshold
        sorted_indices = [
            idx
            for idx in sorted_indices
            if compute_iou(boxes[current], boxes[idx]) < iou_threshold
        ]

    return selected_indices


def smooth_l1_loss(t_predicted: np.ndarray, t_target: np.ndarray) -> float:
    """
    Seção 7.2.2 & 7.2.3: Função de perda Smooth L1 usada no regressor de Bounding Box
    do Fast R-CNN e Faster R-CNN (menos sensível a outliers que a perda L2).
    """
    diff = np.abs(t_predicted - t_target)
    loss = np.where(diff < 1.0, 0.5 * (diff ** 2), diff - 0.5)
    return float(np.sum(loss))


def build_faster_rcnn_rpn_head(feature_map_shape=(14, 14, 512), k_anchors=9) -> Model:
    """
    Seção 7.2.3 (Figura 7.14): Implementação convolucional da Region Proposal Network (RPN).
    Recebe o mapa de características da rede base, aplica CONV 3x3 (512 canais)
    e ramifica em duas camadas CONV 1x1 irmãs:
      - cls_layer: 2*k canais (pontuação objeto vs não-objeto para cada âncora)
      - reg_layer: 4*k canais (deltas dx, dy, dw, dh para cada âncora)
    """
    feature_map_input = Input(shape=feature_map_shape, name="base_feature_map")
    shared_conv = Conv2D(512, (3, 3), padding="same", activation="relu", name="rpn_conv_3x3")(
        feature_map_input
    )

    rpn_cls_scores = Conv2D(
        2 * k_anchors, (1, 1), activation="softmax", name="rpn_cls_2k_scores"
    )(shared_conv)
    rpn_bbox_deltas = Conv2D(
        4 * k_anchors, (1, 1), activation="linear", name="rpn_reg_4k_coords"
    )(shared_conv)

    return Model(
        inputs=feature_map_input,
        outputs=[rpn_cls_scores, rpn_bbox_deltas],
        name="Faster_RCNN_RPN_Head",
    )


def build_ssd_multiscale_layers(pool5_shape=(19, 19, 512)) -> Model:
    """
    Seção 7.3.3 (Página 344): Camadas de características multiescala do SSD300 (conv6 a conv11_2),
    incluindo o uso de Convolução Dilatada (Atrous Convolution) na conv6.
    """
    pool5_input = Input(shape=pool5_shape, name="vgg_pool5_output")

    # conv6 usa convolução dilatada (dilation_rate=(6,6)) e conv7 usa 1x1
    conv6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation="relu", padding="same", name="conv6")(
        pool5_input
    )
    conv7 = Conv2D(1024, (1, 1), activation="relu", padding="same", name="conv7")(conv6)

    # Bloco conv8 (19x19 -> 10x10 usando padding 'same' com stride 2 ou padding customizado)
    conv8_1 = Conv2D(256, (1, 1), activation="relu", padding="same", name="conv8_1")(conv7)
    conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), activation="relu", padding="same", name="conv8_2")(
        conv8_1
    )

    # Bloco conv9 (10x10 -> 5x5)
    conv9_1 = Conv2D(128, (1, 1), activation="relu", padding="same", name="conv9_1")(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation="relu", padding="same", name="conv9_2")(
        conv9_1
    )

    # Bloco conv10 (5x5 -> 3x3 com padding 'valid' e stride 1)
    conv10_1 = Conv2D(128, (1, 1), activation="relu", padding="same", name="conv10_1")(conv9_2)
    conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="valid", name="conv10_2")(
        conv10_1
    )

    # Bloco conv11 (3x3 -> 1x1 com padding 'valid' e stride 1)
    conv11_1 = Conv2D(128, (1, 1), activation="relu", padding="same", name="conv11_1")(conv10_2)
    conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="valid", name="conv11_2")(
        conv11_1
    )

    return Model(
        inputs=pool5_input,
        outputs=[conv7, conv8_2, conv9_2, conv10_2, conv11_2],
        name="SSD_Multiscale_Feature_Layers",
    )


# =====================================================================
# EXECUÇÃO DA SUÍTE DE TESTES PARA DEMONSTRAÇÃO DE DOMÍNIO
# =====================================================================

def run_all_tests():
    print("=" * 78)
    print(" INICIANDO SUÍTE DE VALIDAÇÃO: DEEP LEARNING FOR VISION SYSTEMS (ELGENDY)")
    print("=" * 78)

    # 1. Teste LeNet-5 + Treinamento Rápido Sintético
    print("\n[1/6] Testando LeNet-5 (Seção 5.2)...")
    lenet = build_lenet5()
    assert lenet.count_params() == 61706, f"Esperado 61.706 params, obteve {lenet.count_params()}"
    print(f"  -> LeNet-5 construída com sucesso! Total de parâmetros: {lenet.count_params():,} (Exato com Fig 5.5)")

    # Demonstração de compilação e 1 época de treino com LearningRateScheduler
    X_dummy = np.random.randn(16, 32, 32, 1).astype(np.float32)
    y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, 10, size=(16,)), num_classes=10)
    lenet.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(learning_rate=lenet_lr_schedule(0)),
        metrics=["accuracy"],
    )
    lr_callback = LearningRateScheduler(lenet_lr_schedule, verbose=0)
    lenet.fit(X_dummy, y_dummy, batch_size=8, epochs=1, callbacks=[lr_callback], verbose=0)
    print("  -> Treinamento de teste da LeNet-5 com LR Scheduler concluído com sucesso!")

    # 2. Teste AlexNet
    print("\n[2/6] Testando AlexNet (Seção 5.3)...")
    alexnet = build_alexnet(input_shape=(227, 227, 3), num_classes=1000)
    print(f"  -> AlexNet construída com sucesso! Total de parâmetros: {alexnet.count_params():,}")
    print(f"  -> Shape de saída da AlexNet: {alexnet.output_shape}")

    # 3. Teste VGG-16
    print("\n[3/6] Testando VGG-16 Configuração D (Seção 5.4)...")
    vgg16 = build_vgg16(input_shape=(224, 224, 3), num_classes=1000)
    assert vgg16.count_params() == 138357544, "Contagem de parâmetros da VGG-16 divergente!"
    print(
        f"  -> VGG-16 construída com sucesso! Total de parâmetros: {vgg16.count_params():,} (Exato com Fig 5.14)"
    )

    # 4. Teste GoogLeNet / Inception
    print("\n[4/6] Testando GoogLeNet / Módulo Inception com Gargalo 1x1 (Seção 5.5)...")
    googlenet = build_googlenet(input_shape=(224, 224, 3), num_classes=1000)
    print(
        f"  -> GoogLeNet (9 blocos Inception) construída! Total de parâmetros: {googlenet.count_params():,}"
    )
    print(
        f"  -> Economia de parâmetros vs VGG-16: {138357544 / googlenet.count_params():.1f}x menor!"
    )

    # 5. Teste ResNet-50
    print("\n[5/6] Testando ResNet-50 com Blocos Residuais Bottleneck (Seção 5.6)...")
    resnet50 = build_resnet50(input_shape=(224, 224, 3), classes=1000)
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    pred = resnet50(dummy_img, training=False)
    assert pred.shape == (1, 1000), "Formato de saída da ResNet-50 incorreto!"
    print(
        f"  -> ResNet-50 construída e validada! Total de parâmetros: {resnet50.count_params():,}"
    )

    # 6. Teste dos Componentes de Detecção de Objetos (Capítulo 7)
    print("\n[6/6] Testando Módulos de Detecção: RPN, SSD, YOLOv3, IoU e NMS (Capítulo 7)...")

    # 6.1 RPN Head (Faster R-CNN)
    rpn = build_faster_rcnn_rpn_head(feature_map_shape=(14, 14, 512), k_anchors=9)
    dummy_fmap = np.zeros((1, 14, 14, 512), dtype=np.float32)
    cls_out, reg_out = rpn(dummy_fmap)
    assert cls_out.shape == (1, 14, 14, 18) and reg_out.shape == (1, 14, 14, 36)
    print(
        f"  -> Faster R-CNN RPN Head validada! Saídas p/ k=9 âncoras: cls={cls_out.shape}, reg={reg_out.shape}"
    )

    # 6.2 Camadas Multiescala do SSD + Contagem de 8.732 caixas
    ssd_extra = build_ssd_multiscale_layers(pool5_shape=(19, 19, 512))
    ssd_fmaps = ssd_extra(np.zeros((1, 19, 19, 512), dtype=np.float32))
    ssd_shapes = [(38, 38)] + [(f.shape[1], f.shape[2]) for f in ssd_fmaps]
    priors_per_cell = [4, 6, 6, 6, 4, 4]
    total_ssd_boxes = sum(h * w * p for (h, w), p in zip(ssd_shapes, priors_per_cell))
    assert total_ssd_boxes == 8732
    print(
        f"  -> SSD300 Multiscale Feature Maps: {ssd_shapes} -> Total = {total_ssd_boxes:,} caixas delimitadoras!"
    )

    # 6.3 Contagem de Caixas e Tensor de Saída do YOLOv3
    yolo_grids = [(13, 13), (26, 26), (52, 52)]
    B_anchors = 3
    K_classes = 80  # MS COCO
    total_yolo_boxes = sum(g[0] * g[1] * B_anchors for g in yolo_grids)
    channels_per_scale = B_anchors * (5 + K_classes)
    assert total_yolo_boxes == 10647 and channels_per_scale == 255
    print(
        f"  -> YOLOv3 (416x416): Grades {yolo_grids} * 3 âncoras = {total_yolo_boxes:,} caixas ({channels_per_scale} canais/escala)"
    )

    # 6.4 Teste de IoU, Smooth L1 Loss e Non-Maximum Suppression (NMS)
    gt_box = [100, 100, 50, 50]
    candidate_boxes = [
        [102, 98, 50, 50],   # Caixa 0: Altíssima sobreposição com o objeto (conf 0.95)
        [105, 102, 48, 52],  # Caixa 1: Duplicata redundante sobre o mesmo cachorro (conf 0.82)
        [250, 250, 60, 60],  # Caixa 2: Outro objeto distinto na imagem (conf 0.88)
        [50, 50, 30, 30],    # Caixa 3: Ruído de fundo com baixa confiança (conf 0.25)
    ]
    scores = [0.95, 0.82, 0.88, 0.25]

    iou_0_1 = compute_iou(candidate_boxes[0], candidate_boxes[1])
    kept = non_maximum_suppression(candidate_boxes, scores, conf_threshold=0.5, iou_threshold=0.45)
    sl1 = smooth_l1_loss(np.array(candidate_boxes[0]), np.array(gt_box))

    assert kept == [0, 2], f"NMS falhou! Esperado [0, 2], obteve {kept}"
    print(
        f"  -> Teste Geométrico: IoU(Box0, Box1) = {iou_0_1:.4f} | Smooth L1 Loss = {sl1:.2f} | Caixas mantidas pós-NMS = {kept}"
    )

    print("\n" + "=" * 78)
    print(" TODOS OS TESTES ARQUITETURAIS E MATEMÁTICOS PASSARAM COM 100% DE SUCESSO!")
    print("=" * 78)


if __name__ == "__main__":
    run_all_tests()