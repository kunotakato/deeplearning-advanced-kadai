from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import save_model  
import numpy as np

from .models import UploadedImage
from io import BytesIO
import os
# VGG16モデルのロード
model = VGG16(weights='imagenet')
save_model(model, 'vgg16.h5')  

def predict_image(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())

             #画像のリサイズ
            img = load_img(img_file, target_size=(224, 224))
             # 画像をNumpy配列に変換
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            # 画像の判定
            predictions = model.predict(img_array)
            # 予測結果を解釈し、表示用の結果を生成
            prediction = decode_predictions(predictions, top=5)[0]
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': prediction, 'img_data': img_data})
    else:
        form = ImageUploadForm()

    return render(request, 'home.html', {'form': form})