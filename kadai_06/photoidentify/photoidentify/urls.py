from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static
from prediction.views import predict_image

urlpatterns = [
    path('', include('prediction.urls')),
]

urlpatterns = [
    path('', predict_image, name='predict_image'),  
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)