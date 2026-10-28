# 🧠 Arquiteturas de Deep Learning para Visão Computacional
> Baseado nos Capítulos 5 (*Advanced CNN Architectures*) e 7 (*Object Detection with R-CNN, SSD, and YOLO*) do livro **Deep Learning for Vision Systems** de **Mohamed Elgendy**.

Este repositório documenta a evolução teórica, matemática e prática das principais arquiteturas de Redes Neurais Convolucionais (CNNs) para **Classificação de Imagens** e **Detecção de Objetos**, acompanhado de uma suíte de testes em TensorFlow/Keras para validação de tensores, contagem de parâmetros e algoritmos geométricos de detecção.

---

## 📌 Parte 1: Arquiteturas Avançadas de Classificação (Capítulo 5)

### 1.1 Padrões Universais de Projeto de CNNs
Antes de analisar cada arquitetura individualmente, o livro estabelece três padrões fundamentais que regem quase todas as CNNs:
1. **Padrão #1 (Divisão Funcional):** Toda CNN é dividida em duas partes: **Extrator de Características** (camadas `CONV` + `POOL` empilhadas) seguido pelo **Classificador** (camadas `Fully Connected / Dense` + `Softmax`).
2. **Padrão #2 (Progressão Volumétrica):** A entrada de cada camada é um volume 3D $(H \times W \times D)$. À medida que avançamos na rede, as dimensões espaciais $(H, W)$ **diminuem** (via *pooling* ou *strides* $>1$) enquanto a profundidade $(D$, número de mapas de características$)$ **aumenta**.
3. **Padrão #3 (Camadas Totalmente Conectadas):** O número de unidades ocultas nas camadas densas finais permanece constante (ex: $4096 \rightarrow 4096$) ou diminui a cada camada, raramente aumentando.

> **Conceito-Chave — Camadas de Peso (*Weight Layers*):** Apenas camadas que possuem parâmetros treináveis (`CONV` e `FC/Dense`) são contabilizadas no nome da arquitetura (ex: VGG-16 tem 13 `CONV` + 3 `FC`). Camadas de `Pooling`, `Dropout` e `Activation` não contam como camadas de peso.

---

### 1.2 Evolução das Arquiteturas Clássicas e Avançadas

| Arquitetura | Ano | Camadas de Peso | Parâmetros | Erro Top-5 (ImageNet) | Principal Inovação / Motivação |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LeNet-5** | 1998 | 5 (3 CONV + 2 FC) | ~61,7 mil | >99% Acc (MNIST) | Estrutura pioneira de convolução + subamostragem (*Average Pooling*) + ativação simétrica `tanh`. |
| **AlexNet** | 2012 | 8 (5 CONV + 3 FC) | ~62,3 milhões | 15,3% | Introdução da ativação **ReLU** (combate o *vanishing gradient*), **Dropout (0.5)**, Data Augmentation e treino multi-GPU. |
| **VGG-16** | 2014 | 16 (13 CONV + 3 FC)| ~138,3 milhões | 8,1% (7,4% VGG-19) | **Uniformidade arquitetural:** substitui kernels grandes ($11\times11$, $5\times5$) exclusivamente por blocos de filtros **$3\times3$** ($s=1$) e MaxPool **$2\times2$** ($s=2$). |
| **GoogLeNet (Inception v1)**| 2014 | 22 | ~6,8 a 13 milhões | 6,67% | **Módulo Inception** (extração multiescala em paralelo) + **Camadas Bottleneck ($1\times1$ CONV)** para redução drástica de custo computacional. |
| **ResNet-50 / 152** | 2015 | 50 / 152 | ~25,6M / ~60M | 4,49% (3,57% ensemble)| **Blocos Residuais com Conexões de Salto (*Skip Connections*)**, permitindo treinar redes centenárias sem degradação de gradiente. |

---

### 1.3 Detalhamento Técnico e Matemático das Redes (Cap. 5)

#### 🔹 LeNet-5 (LeCun et al., 1998)
* **Fluxo:** `INPUT (32x32x1) -> C1 (6@5x5, s=1) -> tanh -> S2 (AvgPool 2x2, s=2) -> C3 (16@5x5, s=1) -> tanh -> S4 (AvgPool 2x2, s=2) -> C5 (120@5x5, s=1) -> Flatten -> FC6 (84) -> tanh -> Output (10, Softmax)`.
* **Agendamento de Taxa de Aprendizado (*LR Schedule*):** Otimizador SGD com taxa decaindo por época:
  * Épocas 1–2: $\eta = 0.0005$
  * Épocas 3–5: $\eta = 0.0002$
  * Épocas 6–9: $\eta = 0.00005$
  * Época 10+: $\eta = 0.00001$

#### 🔹 AlexNet (Krizhevsky et al., 2012)
* **Nota sobre Dimensão de Entrada:** Embora o artigo original mencione $224\times224\times3$, a matemática exata da primeira camada com kernel $11\times11$ e stride $4$ exige entrada de **$227\times227\times3$**:
  $$W_{out} = \frac{W_{in} - F + 2P}{S} + 1 = \frac{227 - 11 + 0}{4} + 1 = 55 \implies 55 \times 55 \times 96$$
* **Por que ReLU em vez de Sigmoid/Tanh?** Funções saturadas comprimem entradas grandes em intervalos pequenos ($[0,1]$ ou $[-1,1]$), gerando derivadas próximas de zero nas caudas. A função $\text{ReLU}(x) = \max(0, x)$ mantém gradiente constante igual a $1$ para $x > 0$, acelerando o treino em até 6x.
* **Regularização:** Decaimento de peso L2 ($\lambda = 0.0005$), Dropout de $50\%$ nas camadas FC6 e FC7, e substituição moderna do *Local Response Normalization (LRN)* por **Batch Normalization**.

#### 🔹 VGGNet (Simonyan & Zisserman, 2014)
* **Por que empilhar filtros $3\times3$ é melhor que usar filtros grandes?**
  1. **Mesmo Campo Receptivo Efetivo (*Receptive Field*):** Duas camadas $3\times3$ empilhadas equivalem a um filtro $5\times5$; três camadas $3\times3$ equivalem a um único filtro $7\times7$.
  2. **Mais Não-Linearidade:** Três camadas $3\times3$ aplicam **três funções ReLU** seguidas (contra apenas uma em um filtro $7\times7$), tornando a função de decisão muito mais discriminativa.
  3. **Economia de Parâmetros (81% de redução):** Para $C$ canais, uma única camada $7\times7$ exige $7^2 C^2 = 49C^2$ pesos, enquanto três camadas $3\times3$ exigem apenas $3 \times (3^2 C^2) = 27C^2$ pesos.

#### 🔹 Inception / GoogLeNet (Szegedy et al., 2014)
* **Módulo Inception Ingênuo vs. Com Redução de Dimensionalidade:**
  * Aplicar uma convolução $5\times5$ com $32$ filtros diretamente sobre um volume $32\times32\times200$ exige:
    $$\text{Ops}_{\text{ingênuo}} = (32 \times 32 \times 32) \times (5 \times 5 \times 200) = 163.840.000 \text{ multiplicações (~163,8M)}$$
  * Inserindo uma **Camada de Gargalo (*Bottleneck* $1\times1$ CONV)** com $16$ filtros antes da $5\times5$:
    $$\text{Ops}_{1\times1} = (32 \times 32 \times 16) \times (1 \times 1 \times 200) = 3.276.800 \text{ (~3,2M)}$$
    $$\text{Ops}_{5\times5} = (32 \times 32 \times 32) \times (5 \times 5 \times 16) = 13.107.200 \text{ (~13,1M)}$$
    $$\text{Ops}_{\text{total}} = 3,2\text{M} + 13,1\text{M} = 16,38\text{M operações (Redução de 90% no custo!)}$$
* **Substituição de Camadas FC:** A GoogLeNet substitui as pesadas camadas densas de 4096 neurônios por um **Average Pooling $7\times7$** no final, derrubando o total de parâmetros de 138M (VGG) para ~6,8M–13M.

#### 🔹 ResNet-50 (He et al., 2015)
* **Formulação do Bloco Residual:** Em vez de forçar um bloco de camadas a aprender diretamente um mapeamento subjacente $H(x)$, a rede aprende o resíduo $F(x) = H(x) - x$. A saída antes da ativação final torna-se $F(x) + x$, e a saída do bloco é:
  $$y = \text{ReLU}(F(x, \{W_i\}) + x)$$
* **Bloco Residual com Gargalo (*Bottleneck Residual Block*):** Usa 3 convoluções no caminho principal: `1x1 CONV (reduz canais) -> 3x3 CONV -> 1x1 CONV (expande canais * 4)`.
* **Atalho Regular vs. Atalho de Redução (*Reduce Shortcut*):** Quando o bloco muda a dimensão espacial ($s=2$) ou o número de canais de saída ($F_3$), o caminho de atalho $x$ recebe uma camada `1x1 CONV + BatchNorm` com o mesmo stride $s$ para igualar as dimensões antes da soma matricial `Add()([X, X_shortcut])`.

---

## 🎯 Parte 2: Arquiteturas de Detecção de Objetos (Capítulo 7)

### 2.1 Pipeline Geral e Métricas Fundamentais
Diferente da classificação (que prevê apenas um rótulo $C$), a detecção de objetos localiza e classifica múltiplas instâncias, gerando tuplas $(x, y, w, h, P_{\text{obj}}, C_1, C_2, \dots, C_K)$.

1. **Intersection over Union (IoU):** Mede a sobreposição geométrica entre a caixa real anotada ($B_{\text{gt}}$) e a caixa prevista pelo modelo ($B_{\text{pred}}$):
   $$\text{IoU} = \frac{\text{Área}(B_{\text{gt}} \cap B_{\text{pred}})}{\text{Área}(B_{\text{gt}} \cup B_{\text{pred}})}$$
   * Uma detecção é considerada **Verdadeiro Positivo (TP)** se $\text{IoU} \ge \text{limiar}$ (tipicamente $0.5$ ou $0.75$).
2. **Precisão, Recall e mAP (*mean Average Precision*):**
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
   * Variando o limiar de confiança, plota-se a **Curva Precision-Recall (PR)** para cada classe.
   * **AP (Average Precision):** Área sob a curva PR ($\text{AUC}$) de uma classe específica.
   * **mAP:** Média aritmética do AP entre todas as $K$ classes do dataset.
3. **Non-Maximum Suppression (NMS):** Algoritmo de pós-processamento que elimina detecções duplicadas sobre o mesmo objeto:
   * Passo 1: Descarta caixas com probabilidade abaixo do `confidence_threshold`.
   * Passo 2: Seleciona a caixa restante de maior pontuação como detecção válida.
   * Passo 3: Calcula o IoU dessa caixa contra todas as outras da mesma classe e **suprime (remove)** aquelas cujo IoU seja maior que o `nms_threshold` (ex: $0.45$ ou $0.5$).

---

### 2.2 Comparativo: Detectores de Dois Estágios vs. Estágio Único

| Característica | **R-CNN** (2014) | **Fast R-CNN** (2015) | **Faster R-CNN** (2016) | **SSD300** (2016) | **YOLOv3** (2018) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Família / Tipo** | Two-Stage | Two-Stage | Two-Stage | One-Stage | One-Stage |
| **Proposta de Regiões** | Selective Search (~2.000 ROIs na imagem) | Selective Search (~2.000 ROIs projetadas no Feature Map) | **RPN** (*Region Proposal Network* com *Anchor Boxes*) | Sem RPN (Usa *Priors* densos em 6 mapas multiescala) | Sem RPN (Divide em grades $S \times S$ em 3 escalas) |
| **Backbone Padrão** | AlexNet / VGG (rodada 2.000x por imagem!) | VGG-16 (rodada 1x por imagem + *RoI Pooling*) | VGG-16 / ResNet + RPN compartilhando convoluções | VGG-16 truncada + 6 camadas CONV extras | **Darknet-53** (53 camadas base + 53 detecção = 106) |
| **Classificador Final** | SVMs lineares separadas | Camada Softmax integrada | Camada Softmax integrada | Convoluções $3\times3$ diretas | Convoluções $1\times1$ com **Sigmoid** (multirrótulo) |
| **Velocidade (Teste)** | ~50 seg/img (0.02 FPS) | ~2 seg/img (0.5 FPS) | **0.2 seg/img (5–7 FPS)** | **59 FPS** (SSD300) | **Tempo Real (>30–45 FPS)** |

---

### 2.3 Matemática das Arquiteturas de Detecção

#### 🔹 Função de Perda Multitarefa (Fast / Faster R-CNN)
Combina a perda de classificação logarítmica ($L_{\text{cls}}$) com a regressão das coordenadas de deslocamento da caixa delimitadora ($L_{\text{loc}}$ usando **Smooth L1 Loss**):
$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{\text{cls}}} \sum_i L_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{loc}}} \sum_i p_i^* \cdot L_{1\text{,smooth}}(t_i - t_i^*)$$
* Onde $p_i^* \in \{0, 1\}$ é a função indicadora (igual a $1$ se a âncora for objeto/foreground com $\text{IoU} > 0.7$, e $0$ se for background com $\text{IoU} < 0.3$, garantindo que caixas de fundo não penalizem o regressor espacial).
* Na **RPN**, para $k=9$ âncoras por posição da janela deslizante (3 escalas $\times$ 3 proporções), a camada `1x1 CONV` de classificação gera **$2k = 18$ saídas** (objeto vs. não-objeto) e a camada `1x1 CONV` de regressão gera **$4k = 36$ coordenadas** $(\Delta x, \Delta y, \Delta w, \Delta h)$.

#### 🔹 Cálculo das 8.732 Caixas no SSD (Single Shot MultiBox Detector)
O SSD realiza previsões em **6 mapas de características de escalas decrescentes** (capturando objetos pequenos nas camadas iniciais de alta resolução e objetos grandes nas camadas finais):
* `Conv4_3`: $38 \times 38 \times 4 \text{ priors} = 5.776$
* `Conv7`: $19 \times 19 \times 6 \text{ priors} = 2.166$
* `Conv8_2`: $10 \times 10 \times 6 \text{ priors} = 600$
* `Conv9_2`: $5 \times 5 \times 6 \text{ priors} = 150$
* `Conv10_2`: $3 \times 3 \times 4 \text{ priors} = 36$
* `Conv11_2`: $1 \times 1 \times 4 \text{ priors} = 4$
* **Total:** $5776 + 2166 + 600 + 150 + 36 + 4 = \mathbf{8.732 \text{ caixas delimitadoras por imagem}}$.

#### 🔹 Cálculo das 10.647 Caixas e Tensor de Saída no YOLOv3
Para uma imagem de entrada de $416 \times 416$, o YOLOv3 utiliza *strides* de $32$, $16$ e $8$ (com *upsampling* e concatenação de rotas tipo FPN), gerando grades em 3 escalas ($13\times13$ para objetos grandes, $26\times26$ para médios e $52\times52$ para pequenos), cada célula prevendo $B=3$ caixas:
$$\text{Total de Caixas YOLOv3} = \left[(13 \times 13) + (26 \times 26) + (52 \times 52)\right] \times 3 = (169 + 676 + 2704) \times 3 = \mathbf{10.647 \text{ caixas}}$$
* **Formato do Tensor em cada Escala:** Para $K$ classes, cada célula prevê $B \times (5 + K)$ valores, onde $5$ corresponde a $(t_x, t_y, t_w, t_h, P_o)$.
* **Mudança Crítica no YOLOv3:** Substituiu a ativação `Softmax` por **classificadores independentes com `Sigmoid`** para permitir modelagem **multirrótulo** (ex: um mesmo objeto pode ser simultaneamente *"Pessoa"* e *"Mulher"* sem competição mútua).

---

## 🚀 Como Executar a Suíte de Testes (`test_architectures.py`)

O arquivo `test_architectures.py` implementa, corrige os erros de OCR/tradução do PDF original e executa testes automatizados para todas as arquiteturas dos Capítulos 5 e 7.

### 1. Pré-requisitos
```bash
pip install tensorflow numpy
```

### 2. Executar todos os testes e demonstrações
```bash
python test_architectures.py
```

### 3. O que o script valida na prática?
1. **LeNet-5 + LR Scheduler:** Constrói a rede, verifica exatamente os **61.706 parâmetros** do livro e faz um teste de treinamento rápido com dados sintéticos no formato MNIST ($32\times32\times1$).
2. **AlexNet (BatchNorm + L2):** Valida a matemática da entrada $227\times227\times3$ vs $224\times224\times3$ e verifica os ~62,3 milhões de parâmetros.
3. **VGG-16:** Constrói os 5 blocos uniformes de convoluções $3\times3$ e valida os exatos **138.357.544 parâmetros** da Figura 5.14 do livro.
4. **GoogLeNet (Módulo Inception):** Implementa a função `inception_module` com redução de dimensionalidade ($1\times1$ CONV) usando a API Funcional do Keras e testa o fluxo de tensores.
5. **ResNet-50 Completa:** Implementa o `bottleneck_residual_block` com atalhos regulares e de redução, validando todas as 50 camadas de peso.
6. **Módulos de Detecção (Cap. 7):**
   * Simulador de **RPN Head** ($2k$ scores e $4k$ coordenadas).
   * Verificador matemático das **8.732 caixas do SSD** (com exemplo de *Dilated Convolution*) e das **10.647 caixas do YOLOv3**.
   * Implementação em NumPy puro de **IoU**, **Smooth L1 Loss** e **Non-Maximum Suppression (NMS)** com caso de teste prático.