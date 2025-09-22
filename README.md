# Face ID Demo - OpenCV + Haar Cascade + LBPH

Aplicação local simples para **detecção** e **identificação** facial em tempo real usando webcam.
A proposta atende aos requisitos mínimos: uso de técnica clássica com **OpenCV**, parâmetros **ajustáveis**, exibição na tela com **retângulos**, pequenos **landmarks** de olhos e **identificação** por **LBPH**. O projeto é intencionalmente enxuto.

## Como rodar

### 1. Requisitos
- Python 3.9 ou superior
- Webcam
- Sistema operacional Windows, Linux ou macOS

### 2. Instalação rápida
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

> Importante: o módulo `cv2.face` faz parte do pacote `opencv-contrib-python` que já está listado no requirements. Se você instalar apenas `opencv-python`, **não** terá LBPH.

### 3. Passos
1. **Coletar imagens do usuário**
   ```bash
   python3 src/collect_faces.py --name "SeuNome" --num 1
   ```
   Teclas: `c` ou `space` captura o rosto atual, `q` sai. Ajuste os parâmetros visíveis no overlay ou passe via CLI.

2. **Treinar o reconhecedor LBPH**
   ```bash
   python3 src/train_lbph.py --data-dir data/faces --model-out data/model/lbph_model.yaml --labels-out data/model/labels.json
   ```

3. **Rodar reconhecimento em tempo real**
   ```bash
   python3 src/recognize.py --model data/model/lbph_model.yaml --labels data/model/labels.json
   ```
   Use os **sliders** da janela "Controles" para ajustar:
   - `scale x100`: fator de escala da detecção (ex: 110 equivale a 1.10)
   - `neighbors`: quantidade de vizinhos do Haar Cascade
   - `minSize`: tamanho mínimo do rosto em pixels
   - `lbph_thr`: limiar de aceitação do LBPH. Distâncias **menores** indicam melhor correspondência.

## O que cada parâmetro faz

- **scaleFactor (detector)**: controla o passo da pirâmide de imagem. Valores menores detectam mais, porém ficam mais lentos.
- **minNeighbors (detector)**: aumento reduz falsos positivos e pode perder faces difíceis.
- **minSize (detector)**: ignora faces muito pequenas, útil para evitar ruído.
- **threshold LBPH (reconhecedor)**: distância máxima para aceitar identificação. Se a distância do melhor match for maior que o limiar, a saída vira "Desconhecido".

## Estrutura do repositório
```
face-id-demo/
  data/
    faces/               # imagens coletadas por pessoa
    model/               # modelo e mapeamento de labels
  src/
    collect_faces.py     # coleta de dataset
    train_lbph.py        # treino do LBPH
    recognize.py         # detecção + reconhecimento com sliders
  requirements.txt
  .gitignore
  README.md



## Nota ética - uso de dados faciais
- Coletar rostos apenas com **consentimento** livre e informado.
- Guardar imagens e modelos de forma **local** e protegida. Evitar subir para a nuvem.
- PermitIR **exclusão** dos dados a qualquer momento.
- Evitar vieses: colete imagens diversas de iluminação e expressões.
- Não usar para vigilância ou decisões sensíveis sem análise de risco e base legal.

## Licença
Uso educacional. Sem garantias. Você é responsável pelo cumprimento de leis locais de privacidade.
