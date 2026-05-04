# Switch Vision Monitor

Локальное Windows-приложение для анализа камеры, области экрана или полного экрана через vision-модель Gemma. Приложение показывает чистый ответ модели, ведет визуальную память и сохраняет доказательные кадры локально.

## Ключевые возможности

- локальный рантайм `local-gemma` без обязательных LM Studio или Ollama;
- автоматическая загрузка Gemma GGUF и `llama.cpp` при первом запуске;
- режимы источника: камера, выбранная область, две области, полный экран;
- темный desktop-интерфейс на Tkinter с быстрым стартом, настройками и логами;
- классификатор сцен и библиотека наблюдений в JSON;
- Redis-память для поиска объектов по сохраненным наблюдениям;
- fallback на локальное временное хранилище, если Redis недоступен;
- локальный быстрый детектор пальцев через MediaPipe;
- сборка в один `SwitchVisionMonitor.exe` через PyInstaller.

## Как это работает

1. Приложение получает кадр из камеры или экрана.
2. Кадр отправляется в локальный `llama-server.exe` с Gemma 4 E2B GGUF.
3. Ответ модели парсится, показывается в UI и сохраняется в `output/`.
4. Для режима визуальной памяти наблюдение индексируется в Redis под ключами `vm:*`.
5. По запросу пользователь может найти объект или сцену по сохраненным observations.

## Быстрый старт

```powershell
python -m pip install -r requirements.txt
python app.py
```

При первом запуске `local-gemma` скачает модель и runtime в локальные папки `models/` и `bin/`. Эти файлы не входят в репозиторий: модель весит несколько гигабайт, а runtime и сборки должны оставаться локальными артефактами.

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

Основные переменные окружения:

```powershell
$env:SWITCH_GEMMA_REPO="bartowski/google_gemma-4-E2B-it-GGUF"
$env:SWITCH_GEMMA_MODEL_FILE="google_gemma-4-E2B-it-Q4_K_M.gguf"
$env:SWITCH_GEMMA_MMPROJ_FILE="mmproj-google_gemma-4-E2B-it-f16.gguf"
$env:SWITCH_GEMMA_MAX_TOKENS="1024"
$env:REDIS_URL="redis://localhost:6379/0"
$env:VISION_MEMORY_TTL_SECONDS="86400"
```

## Сборка EXE

```powershell
.\build_exe.ps1
```

Готовый `SwitchVisionMonitor.exe` должен лежать рядом с локальными папками `models/` и `bin/`. Сам EXE не коммитится в Git из-за размера; для передачи демо-сборки используйте GitHub Release внутри приватного репозитория.

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
- `output/`, модели, бинарники и локальный конфиг исключены из Git;
- все основные данные обработки остаются на локальной машине.

## Структура

```text
switch_monitor/
  desktop.py         # основной desktop UI
  engine.py          # capture loop и анализ кадров
  local_gemma.py     # загрузка модели и llama.cpp
  runtimes.py        # local-gemma, LM Studio, Ollama
  vision_memory.py   # Redis/local visual memory
  storage.py         # output layout и файлы состояния
tests/               # unit-тесты
models/README.md     # пояснение по локальным моделям
build_exe.ps1        # сборка Windows EXE
```
