# DigiNova Staj Görev 2

Bu proje, daha önceden Google Colab üzerinde eğitilmiş olan (uçak, insan yüzü ve motosiklet nesnelerini tespit eden) Object Detector modelinin bir FastAPI sunucusu aracılığıyla Docker konteynerinde sunulması amacıyla hazırlanmıştır. Sadece bu üç sınıfa ait nesnelerin tespiti yapılmaktadır.

## Proje İçeriği

- `model/` klasörü içerisinde önceden eğitilmiş model dosyası (`detector.pth`) ve etiket kodlayıcı (`le.pickle`) bulunmaktadır.
- `src/` klasörü içerisinde FastAPI tabanlı sunucu ve inferans kodları bulunmaktadır.
- Uygulama, `docker compose up` ile ARM tabanlı cihazlarda (Apple Silicon) çalışacak şekilde yapılandırılmıştır.

## Kurulum ve Çalıştırma

1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone https://github.com/Abd2023/diginova_staj_gorev_2.git
   cd diginova_staj_gorev_2
   ```

2. Docker üzerinden servisi çalıştırın:
   ```bash
   docker compose up --build
   ```
   *(Not: Eski bir Docker sürümü kullanıyorsanız komutu `docker-compose up --build` şeklinde tire ile yazabilirsiniz.)*

3. Servis `http://localhost:7001` adresinde yayın yapmaya başlayacaktır.

## Test Etme (Inference)

**Servisi ve modeli test etmek için en kolay yöntem Swagger UI arayüzünü kullanmaktır:**
1. Tarayıcınızdan http://localhost:7001/docs adresine gidin.
2. Açılan sayfada `POST /predict` satırına tıklayın.
3. Sağ üst köşedeki **"Try it out"** butonuna basın.
4. Dosya seçme alanından test etmek istediğiniz görseli (uçak, insan yüzü veya motosiklet içeren) yükleyin.
5. **"Execute"** butonuna basın.
6. Hemen aşağıda yer alan **Response body** kısmında sonucun JSON formatında size döndüğünü göreceksiniz.

Alternatif olarak, terminal üzerinden `cURL` veya Postman/Insomnia gibi bir araç kullanarak da `/predict` endpoint'ine `POST` isteği gönderebilirsiniz. 

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

![Screenshot](Screenshot%202026-05-08%20053523.png)
