# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copiar archivos
!mkdir -p data
!cp "/content/drive/MyDrive/Colab Notebooks/content/data/agr_en_train.csv" data/
!cp "/content/drive/MyDrive/Colab Notebooks/content/data/agr_en_dev.csv" data/


