# Switch Vision Monitor

Локальное Windows-приложение для анализа камеры, области экрана или полного экрана через официальную модель Google Gemma. Проект подготовлен под Gemma 4 Good Hackathon: по умолчанию используются конкурсные файлы Kaggle `gemma-4-good-hackathon` и модель `google/gemma-4-E2B-it`.

## Ключевые возможности

- официальный рантайм `official-gemma` на базе Hugging Face Transformers;
- загрузка конкурсных файлов через `kagglehub.competition_download("gemma-4-good-hackathon")`;
- модель по умолчанию: `google/gemma-4-E2B-it`;
- режимы источника: камера, выбранная область, две области, полный экран;
- темный desktop-интерфейс на Tkinter с быстрым стартом, настройками и логами;
- классификатор сцен и библиотека наблюдений в JSON;
- Redis-память для поиска объектов по сохраненным наблюдениям;
- fallback на локальное временное хранилище, если Redis недоступен;
- локальный быстрый детектор пальцев через MediaPipe;
- сборка в один `SwitchVisionMonitor.exe` через PyInstaller.

## Как это работает

1. Приложение получает кадр из камеры или экрана.
2. Перед стартом рантайм проверяет конкурсные файлы через KaggleHub.
3. Кадр отправляется в официальный Transformers-pipeline `image-text-to-text` с моделью `google/gemma-4-E2B-it`.
4. Ответ модели парсится, показывается в UI и сохраняется в `output/`.
5. Для режима визуальной памяти наблюдение индексируется в Redis под ключами `vm:*`.
6. По запросу пользователь может найти объект или сцену по сохраненным observations.

## Быстрый старт

```powershell
python -m pip install -r requirements.txt
python -m switch_monitor.download_model
python app.py
```

Команда загрузки использует ровно конкурсный KaggleHub-вызов:

```python
import kagglehub

# Download latest version
path = kagglehub.competition_download("gemma-4-good-hackathon")

print("Path to competition files:", path)
```

При первом запуске Transformers скачает официальные веса Google в стандартный Hugging Face cache. Веса модели, runtime-данные, локальный конфиг и `output/` не входят в Git.

Для запуска без консоли:

```powershell
pythonw app.pyw
```

Для Redis-памяти:

```powershell
docker compose up -d redis
```

## Конфигурация

Локальные настройки сохраняются в `app_config.json`; этот файл игнорируется Git. Для нового окружения можно скопировать пример:

```powershell
Copy-Item app_config.example.json app_config.json
```

Основная переменная окружения:

```powershell
$env:SWITCH_GEMMA_MODEL_ID="google/gemma-4-E2B-it"
```

Код проверяет, что модель начинается с `google/`. Это сделано специально для конкурсного требования: в дефолтном пути нет community-конвертаций и неофициальных репозиториев моделей.

## Конкурсная подача

Проект позиционируется для направления Safety & Trust / Impact: локальный визуальный помощник для приватных сред с нестабильным интернетом. Он помогает фиксировать наблюдения, риски и контекст без отправки кадров в облачный сервис.

Для финальной подачи нужны:

- публичный Kaggle-отчет до 1500 слов;
- публичный YouTube-ролик до 3 минут;
- публичный репозиторий кода;
- публичная рабочая демоверсия или файлы демо;
- обложка и медиа в Kaggle media gallery.

Дедлайн: 18 мая 2026 года, 23:59 UTC.

## Сборка EXE

```powershell
.\build_exe.ps1
```

Готовый `SwitchVisionMonitor.exe` не коммитится в Git из-за размера; для передачи демо-сборки используйте GitHub Release внутри приватного репозитория.

## Проверка

```powershell
python -m compileall app.py app.pyw switch_monitor tests
python -m unittest discover -s tests
```

Текущий набор покрывает парсинг ответов, storage, runtime-логику, visual memory и desktop entrypoint.

## Приватность

- распознавание личности по лицу не реализовано;
- чувствительные признаки людей не классифицируются;
- полноразмерные кадры для evidence не сохраняются, сохраняются только thumbnails;
- `output/`, локальные модели, бинарники и локальный конфиг исключены из Git;
- все основные данные обработки остаются на локальной машине.

## Структура

```text
switch_monitor/
  desktop.py         # основной desktop UI
  engine.py          # capture loop и анализ кадров
  runtimes.py        # official-gemma и runtime-интерфейсы
  vision_memory.py   # Redis/local visual memory
  storage.py         # output layout и файлы состояния
tests/               # unit-тесты
models/README.md     # локальная папка для пользовательских артефактов, не для Git
build_exe.ps1        # сборка Windows EXE
```
