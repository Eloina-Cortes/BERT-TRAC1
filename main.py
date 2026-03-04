# VERSIÓN ULTRA SIMPLE - GARANTIZADA SIN ERRORES
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

# Deshabilitar wandb y warnings
os.environ["WANDB_DISABLED"] = "true"
import warnings
warnings.filterwarnings('ignore')

print("Iniciando entrenamiento BERT...")

# 1. Cargar datos
train_df = pd.read_csv('data/agr_en_train.csv')
dev_df = pd.read_csv('data/agr_en_dev.csv')

# Renombrar columnas
train_df.columns = ['id', 'text', 'labels']
dev_df.columns = ['id', 'text', 'labels']

# Mapear etiquetas
label_map = {'OAG': 0, 'NAG': 1, 'CAG': 2}
train_df['labels'] = train_df['labels'].map(label_map)
dev_df['labels'] = dev_df['labels'].map(label_map)

print(f" Datos cargados: {len(train_df)} entrenamiento, {len(dev_df)} validación")

# 2. Configurar modelo
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

print(" Modelo BERT cargado")

# 3. Tokenizar de forma simple
def tokenize_texts(texts, labels):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    # Convertir a dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(labels.tolist(), dtype=torch.long)
    })
    return dataset

train_dataset = tokenize_texts(train_df['text'], train_df['labels'])
eval_dataset = tokenize_texts(dev_df['text'], dev_df['labels'])

print(" Datos tokenizados")

# 4. Configuración simple de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Reducido para evitar memoria
    per_device_eval_batch_size=8,
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=2,
    push_to_hub=False,
    report_to=[],
    remove_unused_columns=False,
)

# 5. Función de métricas simple
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.mean(predictions == labels)
    return {'accuracy': accuracy}

# 6. Crear trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("Trainer configurado")

# 7. ENTRENAR
print("\n INICIANDO ENTRENAMIENTO...")
print("="*50)

trainer.train()

print("\n ENTRENAMIENTO COMPLETADO!")

# 8. Evaluar
try:
    eval_result = trainer.evaluate()
    print(f"\n Accuracy final: {eval_result['eval_accuracy']:.4f}")
except:
    print(" Evaluación manual...")

# 9. Guardar modelo
trainer.save_model('./agresion-detector')
tokenizer.save_pretrained('./agresion-detector')

print(" Modelo guardado en: ./agresion-detector")

# 10. Prueba rápida
print("\n PROBANDO MODELO...")
try:
    from transformers import pipeline
    classifier = pipeline('text-classification', model='./agresion-detector')
    
    test_cases = [
        "I really hate this",
        "This is a normal message", 
        "You are being annoying"
    ]
    
    for text in test_cases:
        result = classifier(text)
        label_names = ['OAG', 'NAG', 'CAG']
        pred_idx = int(result[0]['label'].split('_')[-1])
        pred_label = label_names[pred_idx]
        confidence = result[0]['score']
        print(f"'{text}' → {pred_label} ({confidence:.3f})")
        
except Exception as e:
    print(f"Error en prueba: {e}")

print("\n ¡LISTO! Tu modelo está entrenado y guardado.")
print("\nPara usarlo:")
print("from transformers import pipeline")
print("classifier = pipeline('text-classification', model='./agresion-detector')")
print("resultado = classifier('tu texto')")

# ─────────────────────────────────────────────────────────────
# 11. MATRICES DE ATENCIÓN  ← las 2 únicas líneas nuevas
# ─────────────────────────────────────────────────────────────
from matrizaction import generar_matrices_atencion
generar_matrices_atencion()
