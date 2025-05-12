# Traductor de Lengua de Señas Chilena (LSCH)

Este proyecto es una aplicación web hecha con Flask que permite el reconocimiento de señas mediante una cámara, utilizando un modelo entrenado con TensorFlow y MediaPipe.

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/Ardillo23/LSCH-Test.git
cd LSCH-Test

###2. Crear y activar un entorno virtual

python -m venv .venv
.\.venv\Scripts\activate

###3. Instalar dependencias

pip install -r requirements.txt

###4. Colocar el modelo .h5

Asegúrate de tener el archivo senas7.h5 (u otro modelo compatible) dentro de la carpeta model/.

Si no tienes el modelo, puedes solicitarlo al autor o al profesor del curso.

###5. Ejecutar la aplicación

python app.py

Luego abre tu navegador en http://127.0.0.1:5000


#Estructura del proyecto
📁 model/         → Contiene modelos `.h5`
📁 static/        → Estilos e imágenes (incluye placeholder)
📁 templates/     → Archivos HTML
📄 app.py         → Código principal de la app Flask
📄 requirements.txt
📄 README.md


# Tecnologías usadas
Python 3.8+

Flask

TensorFlow

MediaPipe

OpenCV


📬 Autor
Kevin Aguilera – GitHub @Ardillo23