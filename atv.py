import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
# =====================
# ETAPA 1 – AQUISIÇÃO
# =====================
img = cv2.imread(r'C:\Users\phlcs\Documents\grafica\imagem.jpeg')
if img is None:
    print("Erro: imagem não encontrada!")
    exit()
# =====================
# ETAPA 2 – PROCESSAMENTO
# =====================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 100, 200)
# =====================
# ETAPA 3 – HSV (CORES)
# =====================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
# Alterando saturação (exemplo)
s_mod = cv2.add(s, 50)
# =====================
# ETAPA 4 – HISTOGRAMA
# =====================
plt.hist(gray.ravel(), bins=256)
plt.title("Histograma")
plt.xlabel("Intensidade")
plt.ylabel("Quantidade de pixels")
# Interpretação da iluminação
media = np.mean(gray)
if media > 127:
    iluminacao = "Imagem clara"
else:
    iluminacao = "Imagem escura"
print(iluminacao)
# =====================
# ETAPA 5 – BINARIZAÇÃO
# =====================
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# =====================
# ETAPA 6 – IA (YOLO)
# =====================
model = YOLO("yolov8n.pt")
results = model(img)
annotated = results[0].plot()
# Contagem de objetos
contador = 0
print("\nObjetos detectados:")

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        nome = model.names[cls]
        print("-", nome)
        contador += 1

print("Total de objetos:", contador)

# =====================
# ETAPA 7 – EXIBIÇÃO FINAL
# =====================
cv2.imshow("Imagem Original", img)
cv2.imshow("Escala de Cinza", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Bordas", edges)
cv2.imshow("Binarizacao", thresh)
cv2.imshow("Canal H", h)
cv2.imshow("Canal S", s)
cv2.imshow("Canal V", v)
cv2.imshow("Deteccao com IA (YOLO)", annotated)
# Mostra gráfico por último
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()