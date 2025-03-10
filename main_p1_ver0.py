from libs.functions import *
from libs.plot import plotImg

file = 'kodim23.png'

img_og, img_gray = ImgRGB2Gray(file)

img_norm = apply_sigmoid(img_og, alpha=8, delta= -0.2)

plotImg(img_og, img_norm)

print(f"Entropia dentro de una imagen en escala de grises {Entropy(img_og,[ 8, -0.2])}")
print(f"Entropia dentro de una imagen en rango RGB: {Entropy(img_gray, [8, -0.2])}")
