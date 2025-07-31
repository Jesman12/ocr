import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

def histograma(gray):
  h, w = gray.shape[:2]

  mask_bright = gray > 250
  bright_count = np.sum(mask_bright)
  total_pixels = h * w
  bright_ratio = bright_count / total_pixels

  kernel = np.ones((15, 15), np.uint8)
  bright_map = cv2.filter2D(mask_bright.astype(np.uint8), -1, kernel)
  max_local = np.max(bright_map) / 225  
  if True: 
    plt.imshow(bright_map, cmap='hot')
    plt.title('Mapa de concentración de brillo')
    plt.colorbar()
    plt.show()

  print(f"🔍 Detección localizada:")
  print(f"- % píxeles >250: {bright_ratio:.4f}")
  print(f"- Pico de concentración local (máx): {max_local:.2f}")

  if max_local >= 0.98 and bright_ratio >= 0.0005:
    print("Imagen posiblemente con flash (pico muy intenso)")
    return False
  elif bright_ratio >= 0.005:
    print("Imagen posiblemente con flash (porcentaje alto)")
    return False


  elif np.sum(gray < 30) / total_pixels > 0.3:
    print("Imagen subexpuesta")
    return True
  else:
    print("Imagen con buena exposición")
    return True

# Función auxiliar
def probar_histograma_con_imagenes():
  rutas = {
      "Foto": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto.jpg",
      "Foto2": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto2.jpg",
      "Foto3": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto3.jpg",
      "Foto4": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto4.jpg",
      "Foto5": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto5.jpg",
      "Foto6": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto6.jpg",
      "Foto7": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto7.jpg",
      "Foto8": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto8.jpg",
      "Foto9": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto9.jpg",
      "Foto10": "C:/Users/New-DELL/Documents/arcDeEstadia/uploads/foto10.jpg"
  }

  for nombre, ruta in rutas.items():
    print(f"\n--- Evaluando imagen: {nombre} ---")
    image = cv2.imread(ruta)
    if image is None:
      print(f"No se pudo cargar {ruta}")
      continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_images({nombre: gray}, title=f"Imagen en gris: {nombre}")
    resultado = histograma(gray)
    print(f"Resultado de exposición ({nombre}): {'Buena' if resultado else 'Flash/sobreexpuesta'}")

# Función auxiliar que dibuja rectangulos
def rectText (binary, result):
  for res in result:
    # print("res:", res)
    pt0= tuple(map(int,res[0][0]))
    pt1= tuple(map(int,res[0][1]))
    pt2= tuple(map(int,res[0][2]))
    pt3= tuple(map(int,res[0][3]))

    cv2.rectangle(binary, pt0, pt2, (0, 0, 255), 2)
    cv2.circle(binary, pt0, 2, (255,0,0), 2)
    cv2.circle(binary, pt1, 2, (255,0,0), 2)
    cv2.circle(binary, pt2, 2, (255,0,0), 2)
    cv2.circle(binary, pt3, 2, (255,0,0), 2)
  plot_images({"Imagen con rectangulos" : binary}, "Dibujar rectangulos")

# Función del paso 2.1
def apply_gamma_correction(image, gamma=1.0):
  inv_gamma = 1.0 / gamma
  table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
  return cv2.LUT(image, table)

# Función del paso 2 
def adjust_saturation(image, saturation_factor=1.0):
  hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
  hsv_image[...,1] *= saturation_factor
  hsv_image = np.clip(hsv_image, 0, 255)
  return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)

def apply_thresholding(image):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  return {"Original": image, "BINARY": binary, "BINARY_INV": binary_inv}

def process_image(image_path):
  image = cv2.imread(image_path) # 1.- Cargar imagen
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_saturation = adjust_saturation(image_rgb, 0.5) # 2.- Ajustar saturación 
  image_saturation_gamma = apply_gamma_correction(image_saturation, 1.2) #2.1.- Ajustar gamma
  gray = cv2.cvtColor(image_saturation_gamma, cv2.COLOR_BGR2GRAY) # 3.- Convertir a gris

  dictionary = {
    "Original" : image_rgb,
    "+Saturation" : image_saturation,
    "+Gamma" : image_saturation_gamma,
    "+Escala de grises" : gray
  }

  # image_binarys = apply_thresholding(image_saturation_gamma)

  plot_images(dictionary, title="Preprocesamiento de la imagen")
  
  path = "images/gris.jpg"

  cv2.imwrite(path, gray)

  plot_images({"Imagen en escala de grises" : gray},"Imagen gris")

  return path

# Función del paso 7
def detectOCR(image_path, reader):
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  plot_images({"Imagen para OCR" : gray}, "Dentro de la función de OCR")

  results = reader.readtext(gray)
  if not results:
    print("No se detectó texto en la imagen.")
    return None
  rectText(gray, results)

  output = []
  print("Vamos a regresar el texto que tenga una confianza mayor a 0.4")
  for bbox, text, confidence in results:
    if  confidence < 0.4: continue
    print(f"Texto Detectado: {text}")
    print(f"Confianza: {confidence:.2f}")
    output.append({"texto": text, "confianza": confidence})
  return output

def plot_images(image_dict, title="Comparación de Imágenes", cmap="gray"):
  num_images = len(image_dict)
  fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
  if num_images == 1:
    axes = [axes]
  for ax, (t, img) in zip(axes, image_dict.items()):
    ax.imshow(img, cmap=cmap if len(img.shape) == 2 else None)
    ax.set_title(t)
    ax.axis('off')
  plt.suptitle(title)
  plt.show()

# Función del paso 4.1
def angulo_De_Inclinacion (image):
  line_angles = []
  boxes = []
  for (bbox, text, prob) in image:
    (tl, tr, br, bl) = bbox
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    line_angles.append(angle)
    boxes.append(bbox)
  return [np.median(line_angles), boxes]

# Función del paso 6
def recortar (boxes, w, h, rotated):
  if not boxes:
    print("No se detectaron cajas para recortar.")
    return rotated
  all_x = np.concatenate([[p[0] for p in box] for box in boxes])
  all_y = np.concatenate([[p[1] for p in box] for box in boxes])
  x_min = max(0, int(min(all_x)) - 20)
  x_max = min(w, int(max(all_x)) + 20)
  y_min = max(0, int(min(all_y)) - 20)
  y_max = min(h, int(max(all_y)) + 20)

  return rotated[y_min:y_max, x_min:x_max]

#Función del paso 7
def save_image (cropped, reader):
  ubi_ima_recortada = "images/recortada_correcta.png"
  cv2.imwrite(ubi_ima_recortada, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
  print("Archivo guardado como recortada_correcta.png")
  return detectOCR(ubi_ima_recortada, reader)


def mejoramiento(image_path):
  image_bgr = cv2.imread(image_path)

  if not histograma(image_bgr):
    print("Imagen rechazada por sobreexposición")
    return

  # 4.- Detectar orientación
  reader = easyocr.Reader(['es'], gpu=False)
  results = reader.readtext(image_bgr)

  if not results:
    print("No se detectó texto en la imagen con EasyOCR.")
    return image_bgr

  # 4.1 Calcular ángulos de cada línea
  angle_avg, boxes = angulo_De_Inclinacion(results)

  # 5.- Rotar la imagen
  (h, w) = image_bgr.shape[:2]
  center = (w // 2, h // 2)
  print(f"Ángulo detectado en la imagen original: {angle_avg:.2f}")

  if angle_avg == 0:
    cropped = recortar(boxes, w, h, image_bgr)
    plot_images({"Imagen recortada" : cropped}, "Imagen recortada")
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return save_image(binary, reader)

  M_neg = cv2.getRotationMatrix2D(center, -angle_avg, 1.0)
  rotated_neg = cv2.warpAffine(image_bgr, M_neg, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

  M_pos = cv2.getRotationMatrix2D(center, angle_avg, 1.0)
  rotated_pos = cv2.warpAffine(image_bgr, M_pos, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

  plot_images({
    "Original" : image_bgr,
    f"Rotación -{angle_avg:.2f}°": rotated_neg,
    f"Rotación {angle_avg:.2f}°": rotated_pos
    }, "Comparación de imagenes con orientación corregida (+,-)")
  results_neg = reader.readtext(rotated_neg)
  results_pos = reader.readtext(rotated_pos)

  angle_avg_neg, boxes_neg = angulo_De_Inclinacion(results_neg)
  angle_avg_pos, boxes_pos = angulo_De_Inclinacion(results_pos)

  print(f"Rotación negativa {angle_avg_neg}")
  print(f"Rotación positiva {angle_avg_pos}")

  # 6.- Recortar alrededor del texto 
  if angle_avg_neg == 0:
    cropped = recortar(boxes_neg, w, h, rotated_neg)
    plot_images({"Imagen recortada con angulo neg" : cropped}, "Imagen recortada con rotación neg")
  else:
    cropped = recortar(boxes_pos, w, h, rotated_pos)
    plot_images({"Imagen recortada con angulo pos" : cropped}, "Imagen recortada con rotación pos")

  gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

  _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  plot_images({"Imagen binaria": binary}, "Lista para OCR")
  # 7.- Guardar y realizar OCR
  return save_image(binary, reader)

# Agrega tu imagen con el nombre "foto"  en la carpeta "images"o configura el nombre de la imagen en el código segun sea tu necesidad.
image_process = process_image("images/foto.jpg") 

image_plus = mejoramiento(image_process)
