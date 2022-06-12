# Информация по настройке окружения для разработки

- [Начало работы](#начало-работы)
- [Еще заметки](#еще-заметки)
  - [Дополнение для браузерных ноутбуков, на случай, если запуск jupyter сервера выполняется из base среды.](#дополнение-для-браузерных-ноутбуков-на-случай-если-запуск-jupyter-сервера-выполняется-из-base-среды)

> Файлы больше 300 KB храним в [Git LFS](https://git-lfs.github.com/)

## Начало работы

- Устанавливаем `make`
  - Windows:

      Устанавливаем [chocolatey](https://chocolatey.org/install) и устанавливаем `make` с помощью команды:

      ```powershell
      choco install make
      ```

  - Linux:

      ```bash
      sudo apt-get install build-essential
      ```

- Устанавливаем `python 3.10`
  - Windows

      Устанавливаем через [официальный установщик](https://www.python.org/downloads/)

  - Linux (`deadsnakes`)

      ```bash
      sudo apt install python3.10-dev python3.10-venv python3.10
      ```

- Устанавливаем [poetry](https://python-poetry.org/docs/#installation)
  - Windows

      Используйте [официальные инструкции](https://python-poetry.org/docs/#windows-powershell-install-instructions) или команду `powershell`

      ```powershell
      (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
      ```

  - Linux

      ```bash
      make poetry-download
      ```

- Устанавливаем требуемые пакеты и инструменты с помощью команды

    ```bash
    make dev-init
    ```

- Устанавливаем [Git LFS](https://git-lfs.github.com/) и делаем `git lfs pull`, чтобы подтянуть файлы из LFS.

- [Опционально] Для автоматического создания оглавления в ноутбуках настраиваем `nbextension`:
  - `poetry run jupyter contrib nbextension install --user`
  - `poetry run jupyter nbextension enable toc2/main`

  > Для этого расширения требуется зависимость `nbconvert~=5.6.1` (на момент 2021-12-29)

  - Для экспорта ноутбука с ToC используется шаблон команды `poetry run jupyter nbconvert --to html_embed --template toc2 --output-dir ./exports <путь до файла>`
    - Например, `poetry run jupyter nbconvert --to html_embed --template toc2 --output-dir ./exports notebooks/eda/hotel_booking/EDA_Hotel_Bookings.ipynb`

## Еще заметки

**Conda+Python:**

* *На Windows*

```bash
conda create -n adas python=3.7ы
pip install -r requirements.txt
conda install -c pytorch faiss-cpu
```

* *На Linux*

```bash
conda create -n adas python=3.7
pip install -r requirements.txt
conda install -c pytorch faiss-gpu
```

> Согласно [источнику](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda), на Windows **faiss-gpu** недоступен. Надо будет найти альтернативу. <br> Кандидаты: setsimilaritysearch, elasticsearch.


### Дополнение для браузерных ноутбуков, на случай, если запуск jupyter сервера выполняется из base среды.

Добовляет возможность выбрать нужную среду из браузера.

```bash
conda install nb_conda_kernels
```
> Вкладка Kernel -> Change kernel -> Python [conda evn: X]
