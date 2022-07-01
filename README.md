# ADAS System

- [Содержание репозитория](#содержание-репозитория)
- [Клонирование репоизтория](#клонирование-репоизтория)
- [Как начать разработку](#как-начать-разработку)

## Содержание репозитория

- [maddrive_adas](maddrive_adas) - исходники пакета, внутри деление по решаемым задачам.
- [notebooks](notebooks) - ноутбуки для research и проверки кода, внутри делится по задачам.
- [tests](tests) - тесты пакета, запускаются командой `make tests`

## Клонирование репоизтория

> Если у вас в системене установлен [Git LFS](https://git-lfs.github.com/), то рекомендуем, большие файлы там хранятся.

```bash
git clone https://github.com/lsd-maddrive/adas_system
```

## Как начать разработку

Читай в инфе [DEVELOPMENT.md](DEVELOPMENT.md)


## Как использовать:
* Выкачать веса используя `download_models.py`;
* Рассмотреть ноутбуки в `SignDetectorAndClassifier\notebooks`: `DetectorVideoTest` и `COMPOSER`;
* Если нет бинарей `tesseract-ocr`, передавайте `ignore_tesseract=False` в конструктор `EncoderBasedClassifier`;
