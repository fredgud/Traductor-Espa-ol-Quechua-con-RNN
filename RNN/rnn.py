import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import string
import re
import matplotlib.pyplot as plt

# =============================================================================
# 1. DATOS DE EJEMPLO ESPAÑOL-QQUECHUA
# =============================================================================

# Dataset pequeño de ejemplo - UNA SOLA PALABRA POR TRADUCCIÓN
datos_entrenamiento = [
    ("hola", "napaykullayki"),
    ("casa", "wasi"),
    ("agua", "unu"),
    ("comida", "mikhuna"),
    ("familia", "ayllu"),
    ("trabajo", "llankay"),
    ("amigo", "masi"),
    ("sol", "inti"),
    ("luna", "killa"),
    ("hombre", "qhari"),
    ("mujer", "warmi"),
    ("nino", "wawa"),
    ("gracias", "sulpayki"),
    ("adios", "tupananchiskama"),
    ("amor", "munay"),
    ("vida", "kawsay"),
    ("fuego", "nina"),
    ("tierra", "pacha"),
    ("cielo", "hanaq"),
    ("camino", "nan")
]

# =============================================================================
# 2. PREPROCESAMIENTO DE DATOS (SIMPLIFICADO)
# =============================================================================

def preprocesar_texto(texto):
    """Limpia y normaliza el texto"""
    texto = texto.lower().strip()
    texto = re.sub(r'[^\w\s]', '', texto)  # Remover puntuación
    return texto

# Aplicar preprocesamiento
espanol_oraciones = [preprocesar_texto(esp) for esp, que in datos_entrenamiento]
quechua_oraciones = [preprocesar_texto(que) for esp, que in datos_entrenamiento]

print("🔧 Ejemplo de preprocesamiento:")
print(f"  Español: '{espanol_oraciones[0]}'")
print(f"  Quechua: '{quechua_oraciones[0]}'")

# Crear vocabularios
def crear_vocabulario(oraciones):
    vocab = set()
    for oracion in oraciones:
        for palabra in oracion.split():
            vocab.add(palabra)
    return sorted(list(vocab))

vocab_espanol = crear_vocabulario(espanol_oraciones)
vocab_quechua = crear_vocabulario(quechua_oraciones)

print(f"\n📚 Tamaño del vocabulario:")
print(f"  Español: {len(vocab_espanol)} palabras")
print(f"  Quechua: {len(vocab_quechua)} palabras")

# Mapeos palabra a índice e índice a palabra
palabra_a_indice_esp = {palabra: i for i, palabra in enumerate(vocab_espanol)}
indice_a_palabra_esp = {i: palabra for i, palabra in enumerate(vocab_espanol)}

palabra_a_indice_que = {palabra: i for i, palabra in enumerate(vocab_quechua)}
indice_a_palabra_que = {i: palabra for i, palabra in enumerate(vocab_quechua)}

# Parámetros del modelo
TAMANIO_VOCAB_ESP = len(vocab_espanol)
TAMANIO_VOCAB_QUE = len(vocab_quechua)
TAMANIO_EMBEDDING = 32
UNIDADES_RNN = 64
LONGITUD_MAXIMA = 1  # SOLO UNA PALABRA POR ORACIÓN

print(f"\n⚙️ Parámetros del modelo:")
print(f"  Tamaño vocabulario español: {TAMANIO_VOCAB_ESP}")
print(f"  Tamaño vocabulario quechua: {TAMANIO_VOCAB_QUE}")
print(f"  Longitud máxima: {LONGITUD_MAXIMA}")

def tokenizar_y_padding(oraciones, vocabulario, longitud_maxima):
    """Convierte texto a secuencias numéricas con padding"""
    secuencias = []
    for oracion in oraciones:
        secuencia = []
        for palabra in oracion.split():
            secuencia.append(vocabulario.get(palabra, 0))
        # Para una sola palabra
        if len(secuencia) < longitud_maxima:
            secuencia = secuencia + [0] * (longitud_maxima - len(secuencia))
        else:
            secuencia = secuencia[:longitud_maxima]
        secuencias.append(secuencia)
    return np.array(secuencias)

# Preparar datos de entrada y salida
X = tokenizar_y_padding(espanol_oraciones, palabra_a_indice_esp, LONGITUD_MAXIMA)
# Para y, usamos solo la primera palabra de cada traducción quechua
y = tokenizar_y_padding(quechua_oraciones, palabra_a_indice_que, 1)  # SOLO 1 PALABRA

print(f"\n📦 Forma de los datos:")
print(f"  X (español): {X.shape}")
print(f"  y (quechua): {y.shape}")

# Convertir y a one-hot encoding para categorical_crossentropy
y_categorical = tf.keras.utils.to_categorical(y, num_classes=TAMANIO_VOCAB_QUE)

print(f"  y_categorical (one-hot): {y_categorical.shape}")

# =============================================================================
# 3. CONSTRUCCIÓN DEL MODELO RNN (CORREGIDO - DIMENSIONES COINCIDENTES)
# =============================================================================

def crear_modelo_rnn():
    # Modelo secuencia a etiqueta SIMPLIFICADO
    modelo = keras.Sequential([
        # Capa de embedding para español
        layers.Embedding(
            input_dim=TAMANIO_VOCAB_ESP,
            output_dim=TAMANIO_EMBEDDING,
            input_length=LONGITUD_MAXIMA,
            name='embedding_espanol'
        ),
        
        # Capa RNN simple - return_sequences=False para una sola salida
        layers.SimpleRNN(
            UNIDADES_RNN,
            return_sequences=False,  # IMPORTANTE: False para una salida
            name='rnn_layer'
        ),
        
        # Capa densa intermedia
        layers.Dense(32, activation='relu'),
        
        # Capa de salida para quechua - UNA SOLA PALABRA
        layers.Dense(TAMANIO_VOCAB_QUE, activation='softmax', name='salida_quechua')
    ])
    
    return modelo

# Crear y compilar el modelo
print("🧠 Creando modelo RNN...")
modelo = crear_modelo_rnn()

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo creado y compilado!")
print("\n📋 Resumen del modelo:")
modelo.summary()

# =============================================================================
# 4. ENTRENAMIENTO (CORREGIDO)
# =============================================================================

print(f"\n🎯 Verificación final de dimensiones:")
print(f"  X shape: {X.shape}")
print(f"  y_categorical shape: {y_categorical.shape}")

# Callback para early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

print("\n🚀 Comenzando entrenamiento...")

# Entrenar el modelo
historia = modelo.fit(
    X, 
    y_categorical,
    batch_size=4,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print("✅ Entrenamiento completado!")

# =============================================================================
# 5. FUNCIÓN DE TRADUCCIÓN (CORREGIDA)
# =============================================================================

def traducir(oracion_esp):
    """Traduce una palabra del español al quechua"""
    # Preprocesar
    oracion_limpia = preprocesar_texto(oracion_esp)
    
    # Tomar solo la primera palabra
    palabras = oracion_limpia.split()
    if not palabras:
        return "no_entendido"
    
    primera_palabra = palabras[0]
    
    # Tokenizar
    if primera_palabra in palabra_a_indice_esp:
        secuencia = [palabra_a_indice_esp[primera_palabra]]
    else:
        secuencia = [0]  # <unk>
    
    # Padding
    if len(secuencia) < LONGITUD_MAXIMA:
        secuencia = secuencia + [0] * (LONGITUD_MAXIMA - len(secuencia))
    
    secuencia = np.array([secuencia])
    
    # Predecir
    predicciones = modelo.predict(secuencia, verbose=0)
    
    # Obtener la palabra con mayor probabilidad
    indice_predicho = np.argmax(predicciones[0])
    palabra_traducida = indice_a_palabra_que.get(indice_predicho, 'no_entendido')
    
    return palabra_traducida

# =============================================================================
# 6. PRUEBAS Y EVALUACIÓN
# =============================================================================

print("\n" + "="*50)
print("🧪 PRUEBAS DE TRADUCCIÓN")
print("="*50)

# Probar con algunas palabras
oraciones_prueba = ["hola", "casa", "agua", "familia", "gracias", "sol", "luna"]

print("\n📝 Resultados de traducción:")
for palabra in oraciones_prueba:
    traduccion = traducir(palabra)
    print(f"  Español: '{palabra}' → Quechua: '{traduccion}'")

# Calcular precisión
def calcular_precision():
    correctas = 0
    total = len(datos_entrenamiento)
    
    print(f"\n🔍 Evaluando todas las {total} palabras...")
    for i, (esp, que_original) in enumerate(datos_entrenamiento):
        traduccion = traducir(esp)
        que_limpio = preprocesar_texto(que_original)
        
        if traduccion == que_limpio:
            correctas += 1
            print(f"    ✅ '{esp}' -> '{traduccion}'")
        else:
            print(f"    ❌ '{esp}' -> Esperado: '{que_limpio}', Obtenido: '{traduccion}'")
    
    return correctas / total

precision = calcular_precision()
print(f"\n🎯 Precisión del modelo: {precision:.2%}")

# =============================================================================
# 7. GRÁFICAS
# =============================================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historia.history['loss'], label='Pérdida entrenamiento', linewidth=2)
if 'val_loss' in historia.history:
    plt.plot(historia.history['val_loss'], label='Pérdida validación', linewidth=2)
plt.title('Pérdida durante entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(historia.history['accuracy'], label='Precisión entrenamiento', linewidth=2)
if 'val_accuracy' in historia.history:
    plt.plot(historia.history['val_accuracy'], label='Precisión validación', linewidth=2)
plt.title('Precisión durante entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('entrenamiento_traductor.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. GUARDAR MODELO
# =============================================================================

modelo.save('traductor_espanol_quechua_rnn.h5')
print(f"\n💾 Modelo guardado como 'traductor_espanol_quechua_rnn.h5'")

print("\n" + "="*50)
print("🎉 ¡TRADUCTOR ESPAÑOL-QUECHUA COMPLETADO!")
print("="*50)

# Mostrar métricas finales
if historia.history['accuracy']:
    final_acc = historia.history['accuracy'][-1]
    final_loss = historia.history['loss'][-1]
    print(f"📊 Métricas finales - Precisión: {final_acc:.2%}, Pérdida: {final_loss:.4f}")