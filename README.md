# ADAS System

- [Содержание репозитория](#содержание-репозитория)
- [Клонирование репоизтория](#клонирование-репоизтория)
- [Как начать разработку](#как-начать-разработку)

## Содержание репозитория

- [maddrive_adas](maddrive_adas) - исходники пакета, внутри деление по решаемым задачам.
- [notebooks](notebooks) - ноутбуки для research и проверки кода, внутри делится по задачам.
- [tests](tests) - тесты пакета, запускаются командой `make tests`

## Установка пакета в виртуальное окружение

> Для начала рекомендуется настроить виртуальное окружение командой `python3.8 -m venv venv38` и активировать `source ./venv38/bin/activate`.

Установка выполняется командой `pip install git+https://github.com/lsd-maddrive/adas_system#egg=maddrive-adas`

## Клонирование репоизтория

> Если у вас в системене установлен [Git LFS](https://git-lfs.github.com/), то рекомендуем, большие файлы там хранятся.

```bash
git clone https://github.com/lsd-maddrive/adas_system
```

## Как начать разработку

Читай в инфе [DEVELOPMENT.md](DEVELOPMENT.md)
