from transformers import pipeline
classifier = pipeline('text-classification', model='./agresion-detector')

# Textos más claros para cada categoría
textos_test = [
    "I love this project, great work!",           # NAG esperado
    "You idiot, you ruined everything!",          # OAG esperado  
    "Oh sure, like you're so smart...",          # CAG esperado
    "Thanks for helping me today",               # NAG esperado
    "Muslims are terrorists",                    # OAG esperado
    "Whatever, I guess your idea might work"     # CAG esperado
]

for texto in textos_test:
    resultado = classifier(texto)
    print(f"'{texto}' → {resultado}")

    
# Función simple para clasificar agresión
def detectar_agresion(texto):
    resultado = classifier(texto)
    label_names = ['OAG', 'NAG', 'CAG']
    pred_idx = int(resultado[0]['label'].split('_')[-1])
    pred_label = label_names[pred_idx]
    confidence = resultado[0]['score']
    
    return {
        'texto': texto,
        'clasificacion': pred_label,
        'confianza': confidence,
        'es_agresivo': pred_label in ['OAG', 'CAG']
    }

# Ejemplo de uso
print(detectar_agresion("The live feed is getting disconnected again and again. Whats wrong"))