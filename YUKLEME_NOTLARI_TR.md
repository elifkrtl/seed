# Seedify güncellemesi — yükleme notları (2026-10-31)

Bu klasör, https://github.com/elifkrtl/seed deposunun yeni modele göre revize edilmiş hâlidir.
Depoya yüklenmeden önce bu dosya silinebilir (ya da depoda kalabilir; uygulamayı etkilemez).

## Eski sürüme göre değişenler
| Eski (depodaki) | Yeni |
|---|---|
| MobileNetV3-Large (`models/mobilenetv3_large_best.pt`, 17 MB) | Ölçek artırmalı ShuffleNetV2 1.0× (`models/seedify_shufflenetv2_x1_0_scaleaug.pt`, 5.2 MB) |
| Ham fotoğraf doğrudan 320 × 320 px'e küçültülüyordu (arka plan dahil) | Makaledeki işlem hattı: ISNet ile arka plan kaldırma → sabit ölçekli beyaz daire → 224 px |
| Sınıf sırası elle yazılmıştı | Sınıf adları model dosyasından okunuyor (hata riski yok) |
| Her görüntü dört türden birine atanıyordu | Tohum yok / birden fazla tohum / kadraj dışı ve "tanınmayan tohum" uyarıları |
| — | Arka planı kaldırılmış tohum görüntüsü kullanıcıya gösteriliyor |
| `.gitignore` tüm `.pt` dosyalarını dışlıyordu | Yeni model dosyası depoya dahil ediliyor |

## Doğrulama (bu klasördeki kodla çalıştırıldı)
- `seedify_core.py` (bellek iyileştirmeleri dahil son hâli), makalede kullanılan araştırma kodu (`s09_seedify_predict.py`) ile
  26 görüntüde aynı sonucu verdi: 4 örnek, 12 ana set (Note 10+), 6 S21 ve 4 tohumu silinmiş arka plan görüntüsü.
  Durum ve tür 26/26 aynı; sınıf olasılığı ve tanınmayan tohum skoru farkı ≤ 7.2×10⁻⁷. 21 görüntü sınıflandırıldı, 5'i "tanınmayan"
  oldu (4 arka plan görüntüsü ve 1 S21 görüntüsü; araştırma kodu da aynı kararları verdi).
- Daire oluşturmanın yeni (bellek dostu) hâli eskisiyle 17 görüntüde piksel düzeyinde aynı (fark 0).
- Streamlit testi: dört örnek görüntü doğru türe atandı (Cercis %93.2, Ceratonia %80.4, Gleditsia %96.1, Robinia %53.2).
  Robinia örneğinin olasılığı düşük; ölçek artırmalı model bu örnekte daha az emin. İsterseniz daha net bir Robinia örneği seçilebilir.
- İki tohumlu ve boş kağıt görüntüleri reddedildi; DINOv2 indirilemediğinde uygulama çökmeden, uyarıyla çalıştı.

## GitHub'a yükleme
1. Depodan eski dosyaları silin: `models/mobilenetv3_large_best.pt`, `robin_pseudoacacia.jpg`, `cercis_siliquastrum.jpg`,
   `ceratonia_siliqua.jpg`, `gleditsia_triacanthos.jpg` (örnekler artık `examples/` klasöründe).
2. Bu klasördeki her şeyi depo köküne kopyalayın: `app.py`, `seedify_core.py`, `requirements.txt`, `README.md`, `.gitignore`,
   `.streamlit/`, `models/`, `examples/`.
3. `git add -A && git commit -m "Seedify: background-free pipeline, ShuffleNetV2 scale-aug model, unrecognised-seed check" && git push`

## Streamlit Cloud
- Main file: `app.py`; Python sürümü: 3.11 önerilir (test edilen sürüm).
- İlk açılışta ISNet (~179 MB) ve DINOv2 (~88 MB) ağırlıkları indirilir; ilk yükleme 1–2 dakika sürebilir, sonrası önbellekten gelir.
- Bellek (Mac'te ölçüldü, bir fotoğraf işlenirken en yüksek değer): tanınmayan tohum kontrolü açıkken ≈ 3.0 GB, kapalıyken ≈ 2.9 GB.
  Belleğin çoğu ISNet'in 1024 px girdide çalışmasından gelir; DINOv2 yalnızca ≈ 0.2 GB ekler. onnxruntime bellek havuzu kapatılarak ve
  daire yalnızca tohum çevresinde oluşturularak ilk ölçülen 3.9 GB düşürüldü (sonuçlar piksel düzeyinde aynı).
  Ücretsiz Streamlit Community Cloud'un bellek sınırı bunun altındaysa uygulama "Resource limits" hatasıyla kapanabilir. Bu durumda
  seçenekler: (a) Streamlit hesabında daha yüksek kaynak, (b) daha fazla bellek sunan başka bir barındırma (ör. Hugging Face Spaces
  ücretsiz CPU), (c) ISNet girdi boyutunu küçültmek — bu son seçenek makaledeki yöntemi değiştirir ve yeniden doğrulama gerektirir.
  Tanınmayan tohum kontrolünü `ood_dir=None` ile kapatmak belleği anlamlı ölçüde azaltmaz.
- `requirements.txt` CPU sürümü torch indirir (daha küçük kurulum).

## Kaynak dosyalar (LEGUMINOSE içinde)
- Model: `ANALIZ_ONLY_SEED/v07_olcek_artirma/seedify_shufflenetv2_x1_0_only_seed_scaleaug.pt`
- Tanınmayan tohum bankası ve eşikler: `ANALIZ_ONLY_SEED/v08_tanimsiz_tohum/ood_bank.npz`, `ood_threshold.json`
