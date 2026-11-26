# Casos de energía eólica (material del módulo 16)

Cinco cuadernos en español que acompañan al módulo 16 de Redes Neuronales 1002, *Real Use Cases in Wind Energy*. Cada uno prueba una tecnología de redes neuronales sobre un problema real del sector eólico y compara con un método clásico:

| Cuaderno | Tecnología | Problema |
|---|---|---|
| `01_pinn_perfil_vertical_viento.ipynb` | PINN (red neuronal informada por la física) | perfil vertical del viento con la teoría de Monin-Obukhov |
| `02_gnn_granja_eolica_estelas.ipynb` | GNN (red neuronal de grafos) | potencia de un parque con estelas entre turbinas |
| `03_rosa_espectral_oleaje.ipynb` | procesado de señal y espectros (sin red neuronal) | rosa espectral del oleaje desde boya e imágenes SAR |
| `04_xgboost_vs_pytorch.ipynb` | XGBoost frente a redes en PyTorch | tres problemas tabulares y de series con datos reales |
| `05_cnn_oleaje_sar.ipynb` | CNN (red convolucional) | altura, periodo y dirección del oleaje desde imágenes SAR, validado con ERA5 |

Los datos pequeños que usan están en `datos/`. Se ejecutan desde esta carpeta (`jupyter lab`), con las dependencias de `../requirements.txt` más `pandas`, `xgboost`, `scipy`, `pyarrow`.

**Licencias de los datos (leer antes de reutilizar o publicar).**

Licencias: `datos/sar_sitio_2016.npz` (8 imágenes de enero de 2016 del mismo archivo) y `datos/sar_sitio_2016_espectros.npz` (espectros derivados de 187 imágenes de 2016, sin las imágenes) tienen la misma licencia CC BY-NC-SA 4.0, y `datos/era5_sitio_2016.csv`, `datos/era5_sitio_2016_imagenes.csv` y `datos/era5_serie_sitio_2016.csv` contienen valores interpolados de ERA5 (Copernicus Climate Change Service; licencia de Copernicus, con atribución). El extracto `datos/sar_s1_wv_extracto.npz` (viñetas de TenGeoP-SARwv, doi:10.17882/56796, derivadas de datos Copernicus Sentinel-1 de 2016) es CC BY-NC-SA 4.0: solo uso no comercial y compartir igual. El archivo del lidar EPL no declara licencia; revisar los términos del portal de origen antes de un uso comercial o de redistribuir el extracto.


El texto y el código de los cuadernos son del curso (licencia del repositorio); los datos conservan la licencia de su origen, y varias son de **uso no comercial**.
