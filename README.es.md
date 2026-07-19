# CHTN (Capybla Harmonic Transmission)

> **Formato de audio paramétrico experimental basado en síntesis aditiva en tiempo real.**

![Estado](https://img.shields.io/badge/estado-experimental-blue)
![Lenguaje](https://img.shields.io/badge/python-3.12+-green)
![Licencia](https://img.shields.io/badge/licencia-No_Comercial-red)

---

# Descripción

CHTN es un formato de audio experimental que almacena una **representación paramétrica del sonido** en lugar de guardar directamente la forma de onda.

En lugar de comprimir muestras de audio como hacen MP3 o AAC, CHTN analiza el contenido espectral del sonido y almacena únicamente la información necesaria para reconstruirlo en tiempo real mediante un motor DSP.

El proyecto incluye:

- CHTN Studio
- Codificador
- Decodificador
- Motor DSP en tiempo real
- Asignación dinámica de osciladores
- Visualizadores de audio
- Contenedor CHTN

---

# ¿Cómo funciona?

Los formatos tradicionales almacenan muestras comprimidas.

CHTN sigue un enfoque diferente.

```
Audio Original
      │
      ▼
Análisis Espectral
      │
      ▼
Parámetros de Osciladores y Ruido
      │
      ▼
Archivo .CHTN
      │
      ▼
Motor DSP
      │
      ▼
Audio Reconstruido
```

En lugar de almacenar millones de muestras de audio, CHTN guarda las instrucciones necesarias para reconstruir el sonido.

---

# Características

- Síntesis aditiva en tiempo real
- Osciladores dinámicos
- Representación paramétrica del audio
- Motor multihilo
- Síntesis de ruido
- Audio estéreo
- Visualizadores
- Compatibilidad entre versiones del formato
- Compresión adaptativa experimental

---

# Estado del proyecto

Versión actual:

```
CHTN Studio v31
```

Actualmente permite:

- Reproducir archivos CHTN
- Convertir audio a CHTN
- Decodificación en tiempo real
- Interfaz gráfica completa
- Compatibilidad con:

```
MP3
WAV
FLAC
OGG
M4A
```

---

# Organización del proyecto

```
CHTN/
│
├── engine/
├── studio/
├── docs/
├── examples/
├── images/
├── README.md
├── README.es.md
└── LICENSE
```

---

# Compresión

CHTN **no utiliza un bitrate fijo**.

El tamaño del archivo depende directamente de la complejidad espectral del audio.

Comportamiento habitual:

| Tipo de audio | Compresión |
|---------------|-----------|
| Voz | Excelente |
| Chiptune | Excelente |
| Piano | Muy buena |
| Orquesta | Buena |
| Rock | Moderada |
| Ruido blanco | Baja |

---

# Rendimiento

El motor está diseñado para reconstruir audio en tiempo real.

El uso de CPU depende principalmente de:

- Número de osciladores activos
- Complejidad del audio
- Tamaño del bloque DSP
- Potencia del procesador

La calidad aumenta conforme se incrementa el número máximo de osciladores.

---

# Requisitos

Python 3.12 o superior.

Bibliotecas utilizadas:

- numpy
- scipy
- librosa
- sounddevice
- tkinter
- matplotlib

---

# Filosofía

CHTN nace como un experimento para explorar una forma diferente de representar el audio digital.

En lugar de preguntarse:

> "¿Cómo puedo comprimir mejor una forma de onda?"

la pregunta es:

> "¿Cómo puedo describir matemáticamente el sonido para volver a generarlo?"

El objetivo del proyecto no es sustituir a los formatos actuales, sino investigar nuevas posibilidades para la representación paramétrica del audio.

---

# Hoja de ruta

Próximas mejoras:

- Mejor reconstrucción de frecuencias graves
- Optimización del contenedor CHTN
- Nuevos visualizadores
- Más optimizaciones DSP
- Documentación técnica completa
- Especificación oficial del formato
- Compatibilidad multiplataforma

---

# Licencia

Este proyecto puede utilizarse gratuitamente para fines personales, educativos y de investigación.

Se permite:

- Utilizar
- Estudiar
- Modificar
- Compartir

Es obligatorio:

- Mantener el crédito al autor original.

No se permite el uso comercial sin autorización expresa del autor.

Consulta el archivo LICENSE para obtener todos los detalles.

---

# Autor

**David Hernández (Capybla)**

GitHub:

https://github.com/Capybla

---

*CHTN es un proyecto experimental y continúa en desarrollo.*
