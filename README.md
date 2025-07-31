# Proyecto CARI

## Descripción
Este proyecto es un script en Python que realiza reconocimiento óptico de caracteres (OCR) en imágenes utilizando la librería EasyOCR. Incluye procesamiento previo de imágenes para mejorar la calidad y precisión del OCR, como ajuste de saturación, corrección gamma, conversión a escala de grises, umbralización, corrección de orientación, recorte, binarización, aplicación de OCR para detección de texto y extracción para la construcción de un diccionario.

## Características
- Análisis de exposición de la imagen para detectar sobreexposición o subexposición.
- Ajuste de saturación y corrección gamma para mejorar la calidad de la imagen.
- Detección y corrección de la inclinación del texto en la imagen.
- Recorte automático alrededor de las regiones de texto detectadas.
- Uso de EasyOCR para extraer texto con un umbral de confianza configurable.
- Visualización de imágenes en diferentes etapas del procesamiento.

## Requisitos
- Python 3.10  
- OpenCV (cv2)  
- EasyOCR  
- NumPy  
- Matplotlib  

## Instalación
Instala las dependencias necesarias usando pip:

bash
pip install opencv-python easyocr numpy matplotlib


## Uso
Coloca la imagen que deseas procesar en la carpeta images/ o modifica la ruta en el script.

Ejecuta el script easy_ocr.py:

bash
python easy_ocr.py


El script procesará la imagen, realizará OCR y mostrará los resultados en la consola y mediante gráficos.

## Funciones principales
- histograma(gray): Analiza la exposición de la imagen en escala de grises.
- process_image(image_path): Realiza el preprocesamiento de la imagen (saturación, gamma, escala de grises).
- mejoramiento(image_path): Aplica corrección de orientación, recorte y realiza OCR con EasyOCR.
- detectOCR(image_path, reader): Ejecuta el OCR y devuelve el texto detectado con su confianza.

## Notas
- Se hace uso de la versión 3.10 de Python, porque en versiones superiores tendia a fallar.
- El script está configurado para procesar imágenes en español ('es') con EasyOCR y utiliza CPU, aunque sería más rápido con uso de GPU.
- Se recomienda usar imágenes con buena exposición para obtener mejores resultados.
- La visualización de imágenes es útil para entender cada paso del procesamiento.
- Algunas líneas de código (como las visualizaciones intermedias) pueden eliminarse si no son necesarias.

## Funciones auxiliares
- rectText(binary, result): Dibuja rectángulos alrededor de los textos para mostrar visualmente lo que detecta el OCR.
- probar_histograma_con_imagenes(): Realiza un análisis básico de exposición para verificar si la imagen está bien iluminada.

## Licencia
Este proyecto es de código abierto y puede ser modificado y distribuido libremente.
