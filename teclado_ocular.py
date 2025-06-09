import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image

# Diccionario ampliado y ordenado alfabéticamente
palabras = [
    "adios", "amor", "amigo", "auto", "avanzar", "avion", "azul",
    "bajo", "banco", "barco", "barrio", "bebe", "bien", "bueno",
    "casa", "cielo", "coche", "color", "comer", "comida", "corazon", "correr",
    "decir", "dedo", "deporte", "dia", "dinero", "dormir", "dulce",
    "escribir", "escuela", "espejo", "espacio", "estrella", "estudio",
    "facil", "familia", "feliz", "fiesta", "fuego", "futuro",
    "gato", "gente", "globo", "grande", "gritar", "grupo",
    "hablamos", "habito", "habitacion", "hablar", "hacer", "habil",
    "hambre", "hamaca", "helado", "hermano", "hermoso", "heroina",
    "hola", "holanda", "hotel", "hoy", "huevo", "humano",
    "idea", "iglesia", "invierno", "isla",
    "joven", "juego", "jugar",
    "lago", "libro", "lento", "luz",
    "madre", "mano", "mar", "mañana",
    "niño", "noche", "nuevo", "numero",
    "ojo", "oro", "oso",
    "padre", "palabra", "paz", "perro", "persona", "plaza", "playa",
    "que", "querer", "quizas",
    "raton", "rey", "ropa",
    "saber", "salud", "sol", "sueño",
    "tarde", "taza", "tiempo", "tierra",
    "universo", "usar",
    "verde", "vida", "viento", "vino",
    "voz", "vuelo",
    "zapato", "zanahoria", "zarpar"
]

teclas = [
    list("qwertyuiop"),
    list("asdfghjklñ"),
    list("zxcvbnm ,.")
]

texto = ""
palabra_actual = ""

TIEMPO_SELECCION = 1.5
ultima_fila, ultima_col = -1, -1
ultimo_tiempo = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
IRIS_DERECHO_ID = [474, 475, 476, 477]

sugerencia_x1, sugerencia_y1 = 10, 70
sugerencia_x2, sugerencia_y2 = 600, 110
fijacion_sugerencia_tiempo = 1.5
ultimo_tiempo_sugerencia = 0
mirada_en_sugerencia = False

# Fuente que soporte Unicode
font = ImageFont.truetype("arial.ttf", 32)

def autocompletar(texto_completo):
    palabras_escritas = texto_completo.strip().split(" ")
    if not palabras_escritas:
        return ""
    prefijo = palabras_escritas[-1].lower()
    if not prefijo:
        return ""
    sugerencias = [p for p in palabras if p.startswith(prefijo) and p != prefijo]
    return sugerencias[0] if sugerencias else ""

def detectar_mirada_por_iris(landmarks, img_w, img_h):
    iris_coords = []
    for idx in IRIS_DERECHO_ID:
        x = int(landmarks[idx].x * img_w)
        y = int(landmarks[idx].y * img_h)
        iris_coords.append((x, y))
    if iris_coords:
        iris_center = np.mean(iris_coords, axis=0).astype(int)
        return tuple(iris_center)
    return None

def detectar_tecla_seleccionada(img, x, y):
    h, w = img.shape[:2]
    tecla_w = w // 10
    tecla_h = 60
    y_inicio = h - tecla_h * 3 - 20
    if y < y_inicio:
        return -1, -1
    fila = (y - y_inicio) // tecla_h
    col = x // tecla_w
    if fila < 0 or fila > 2 or col < 0 or col > 9:
        return -1, -1
    return int(fila), int(col)

def dibujar_teclado(img, fila_sel, col_sel):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    h, w = img.shape[:2]
    tecla_w = w // 10
    tecla_h = 60
    y_inicio = h - tecla_h * 3 - 20
    for i, fila in enumerate(teclas):
        for j, letra in enumerate(fila):
            x = j * tecla_w
            y = y_inicio + i * tecla_h
            color = (50, 30, 30) if (i != fila_sel or j != col_sel) else (0, 255, 0)
            draw.rectangle([x, y, x + tecla_w - 2, y + tecla_h - 2], fill=color)
            bbox = draw.textbbox((x, y), letra, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x + (tecla_w - text_w) // 2, y + (tecla_h - text_h) // 2), letra, font=font, fill=(255, 255, 255))
    return np.array(pil_img)

def mostrar_texto_con_pil(img, texto, sugerencia):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle((sugerencia_x1, sugerencia_y1 - 30, sugerencia_x2, sugerencia_y2), outline=(50, 50, 50), width=2)
    draw.text((sugerencia_x1 + 10, sugerencia_y1), f"Sugerencia: {sugerencia}", font=font, fill=(180, 180, 180))
    draw.text((10, 10), f"Texto: {texto}", font=font, fill=(255, 255, 255))
    return np.array(pil_img)

def main():
    global texto, palabra_actual, ultima_fila, ultima_col, ultimo_tiempo
    global mirada_en_sugerencia, ultimo_tiempo_sugerencia

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        fila_sel, col_sel = -1, -1
        iris_pos = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                iris_pos = detectar_mirada_por_iris(face_landmarks.landmark, w, h)
                if iris_pos:
                    cv2.circle(frame, iris_pos, 5, (255, 0, 255), -1)

                    fila_sel, col_sel = detectar_tecla_seleccionada(frame, iris_pos[0], iris_pos[1])

                    x_iris, y_iris = iris_pos
                    if (sugerencia_x1 <= x_iris <= sugerencia_x2) and (sugerencia_y1 - 30 <= y_iris <= sugerencia_y2):
                        if not mirada_en_sugerencia:
                            mirada_en_sugerencia = True
                            ultimo_tiempo_sugerencia = time.time()
                        else:
                            if time.time() - ultimo_tiempo_sugerencia >= fijacion_sugerencia_tiempo:
                                sugerencia = autocompletar(texto)
                                if sugerencia:
                                    texto += sugerencia[len(palabra_actual):] + " "
                                    palabra_actual = ""
                                mirada_en_sugerencia = False
                    else:
                        mirada_en_sugerencia = False

        if fila_sel != -1 and col_sel != -1:
            if (fila_sel, col_sel) == (ultima_fila, ultima_col):
                if time.time() - ultimo_tiempo >= TIEMPO_SELECCION:
                    letra = teclas[fila_sel][col_sel]
                    if letra in [",", ".", " "]:
                        texto += letra
                        palabra_actual = ""
                    else:
                        texto += letra
                        palabra_actual += letra
                    ultimo_tiempo = time.time() + 1
            else:
                ultima_fila, ultima_col = fila_sel, col_sel
                ultimo_tiempo = time.time()

        sugerencia = autocompletar(texto)
        frame = dibujar_teclado(frame, fila_sel, col_sel)
        frame = mostrar_texto_con_pil(frame, texto, sugerencia)

        cv2.imshow("Teclado Virtual con Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
