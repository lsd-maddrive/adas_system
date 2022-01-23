# ADAS System

### Sign Detection&Recognition

# Содержание репозитория

* data - папка для датосетов, .csv и прочего(?), необходимого для обучения. Ноутбуки сами заполняют папку, в случае, если запущены из Google Colab.
* docs - доп. ознакомительные доки.
* notebooks - папка с ноутбуками. Ее необходимо сделать домашней директорией при запуске

        git clone https://github.com/lsd-maddrive/adas_system
        cd adas_system/noteboos
> Внутри notebooks в папке nt_helper вспомогательные файлы для работы ноутбуков

# Описание ноутбуков
* 1_ClassifierResearch - классификатор resnet18 для распознования знаков. В ходе обучения сохраняет веса в data/resnet18_rtsd_test. (<b>FIX ME</b>)
* 2_YoloDetection - детектор на основе YoloV5. В ходе обучения сохраняет веса в notebooks/YoloV5Last.pt. При обучении в Google Colab сохраняет итерации обучения в корень гугл диска.
* VideoTest - использует YoloV5 для демонстрации на видео с регистратора (data/reg_videos/1.mp4)
* RTSD-R_MERGED - создает удобный формат из RTDS. Вероятно не запуститься :(

# Как запустить
## Для ноутбуков 1_ClassifierResearch или 2_YoloDetection
В ноутбуках 1 или 2 исправить флаг

    SHOULD_I_TRAIN = True для запуска обучения,
                     False для теста, загрузки имеющихся весов

Вкладка TEST MODEL позволяет вызвать модель на произвольные данные.

## Для ноутбука VideoTest
Запустить ноутбук целиком, для проигрывания тестового видео с демонстрацией работы детектора




# TO DO
- [x] Softmax confidence для классификатора
- [ ] Метрики по классификатору
- [x] Расположение весов классификатора перенести
- [ ] Рефактор датасетов
- [ ] Метрики детектора
- [ ] DeepSort для детектора
- [ ] Рефактор кода
- [ ] Поддержка TPU? uoss.py err



# Используемые датасеты
| Название | Описание | Источник |
|-|-|-|
| RTSD Public | Состоит из нескольких частей, включая "full-frames" -  размеченные кадры с видеорегистратора. Весит около 18 гб. К нему прилогается csv с координатами знаков. Однаков, не все изображения размечены. Ноутбук детектора скачает укороченную версию (около 6 гб, что позволит спокойно использовать ее в Google Colab) этой части датасета, содержащую только размеченные данные; "detection" - датасет для детекции вобще всего, включая края дороги; "classification" - датасет для классификации знаков | [Ссылка](https://disk.yandex.ru/d/TX5k2hkEm9wqZ) <br /> <br /> [Источник ссылки](https://github.com/sqrlfirst/traffic_sign_work) |
| GTSRB *Recognition* | Немецкий набор знаков, в случае нехватки буду брать отсюда | [Ссылка](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) |
| GTSDB *Detection* | Аналогично предыдущему | [Ссылка](https://www.kaggle.com/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb) |



>Датасеты пакуются в pandas.DataFrame, который является входом DataLoader'ов моделей


# Пример работы:
### Классификатор примеры
Пример 1 | Пример 2
:-------------------------:|:-------------------------:
![alt-text-1](./screenshots/classifier1.png) | ![alt-text-2](./screenshots/classifier2.png)

>Красным выделены неверно классифицированные знаки

### Детектор примеры
Пример 1 | Пример 2
:-------------------------:|:-------------------------:
![alt-text-1](./screenshots/detector1.png) | ![alt-text-2](./screenshots/detector2.png)
