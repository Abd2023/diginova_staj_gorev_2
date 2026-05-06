# DigiNova Staj Görev 2

Bu proje, daha önceden Google Colab üzerinde eğitilmiş olan Object Detector modelinin bir FastAPI sunucusu aracılığıyla Docker konteynerinde sunulması amacıyla hazırlanmıştır. 

## Proje İçeriği

- `model/` klasörü içerisinde önceden eğitilmiş model dosyası (`detector.pth`) ve etiket kodlayıcı (`le.pickle`) bulunmaktadır.
- `src/` klasörü içerisinde FastAPI tabanlı sunucu ve inferans kodları bulunmaktadır.
- Uygulama, `docker-compose up` ile ARM tabanlı cihazlarda (Apple Silicon) çalışacak şekilde yapılandırılmıştır.

## Kurulum ve Çalıştırma

1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone https://github.com/Abd2023/diginova_staj_gorev_2.git
   cd diginova_staj_gorev_2
   ```

2. Docker üzerinden servisi çalıştırın:
   ```bash
   docker-compose up --build
   ```

3. Servis `http://localhost:7001` adresinde yayın yapmaya başlayacaktır.

## Test Etme (Inference)

Aşağıdaki komutu veya Postman/Insomnia gibi bir aracı kullanarak `/predict` endpoint'ine `POST` isteği gönderebilirsiniz. 

Örnek bir cURL komutu:
```bash
curl -X POST http://localhost:7001/predict -F "file=@ornek_gorsel.jpg"
```

### Beklenen Çıktı

Dönen JSON sonucu sınıf etiketini, güven skorunu ve bounding box koordinatlarını içerecektir:
```json
{
  "label": "airplane",
  "confidence": 0.9876,
  "bounding_box": {
    "startX": 0.12,
    "startY": 0.05,
    "endX": 0.89,
    "endY": 0.95
  }
}
```

## Geliştirme Notları

Bu repoda sadece çıkarım (inference) aşaması için gerekli olan dosyalar bulundurulmuştur. Modelin eğitimiyle ilgili kaynak kodlar veya Jupyter dosyaları yer almamaktadır.
