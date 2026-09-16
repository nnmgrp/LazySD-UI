# QuantFunc WebUI

Локальный WebUI для **QuantFunc Engine** — закрытого C++/CUDA движка диффузионного вывода.
Построен на **FastAPI + uvicorn** с полностью кастомным HTML/CSS/JS интерфейсом
(тёмная тема, IDE-grid layout).

**Не требует ComfyUI и не требует Gradio.** Использует QuantFunc engine binary напрямую
через subprocess CLI. UI построен по образцу HF Space
`prithivMLmods/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast` — hand-written `index.html`
+ JSON API endpoints.

## Возможности (Phase 2.5)

- **Text-to-Image**: Flux.2 Klein 9B, Flux.2 Klein 4B, Z-Image Turbo (Lighting-4steps)
- **API-ключ QuantFunc**: автоматическая установка `bin/<os>/config.json` с общим
  бесплатным тестовым ключом (public beta), поле для своего ключа в Settings,
  передача `--apikey` в engine, понятные сообщения об ошибках key-protected моделей
- **GPU detection** через nvidia-smi → automatic SM capability → tier 50x-above / 50x-below
- **Lazy download** через HuggingFace или ModelScope SDK (resume, progress)
- **Cache с configurable path** (settings.json)
- **SHA-256 verify** для engine binary + model weights (gap в plugin закрыт)
- **Bilingual UI** RU/EN с auto-detect по navigator.language
- **Offline-first**: auto-update отключён, ручная проверка через Settings → Check for updates
- **Subprocess worker** с crash isolation и auto-restart

## Установка

### Требования

| Компонент | Версия |
|-----------|--------|
| Python | 3.11+ (рекомендуется 3.11) |
| NVIDIA GPU | RTX 20+ (SM 7.5+) |
| NVIDIA Driver | ≥ 560 |
| CUDA Runtime | 12.x или 13.x |
| cuDNN | 9.x |
| ОС | Windows 10/11 (основной target), Linux (best-effort) |

### Установка на Windows

1. Установите Python 3.11+ с https://python.org (обязательно отметьте "Add Python to PATH")
2. Установите CUDA Toolkit 12.x или 13.x с https://developer.nvidia.com/cuda-toolkit
3. Установите cuDNN 9.x с https://developer.nvidia.com/cudnn
4. Клонируйте или распакуйте этот репозиторий
5. Дважды кликните `start.bat` — он установит зависимости и запустит WebUI

### Ручной запуск

```bash
cd QuantFunc-WebUI
python -m pip install -r requirements.txt
python app.py
```

После запуска откройте в браузере: **http://127.0.0.1:7860**

## Первый запуск

При первом запуске WebUI автоматически:

1. Определит вашу GPU (имя, SM, VRAM)
2. Проверит наличие engine binary `libquantfunc.so` / `quantfunc.dll`
3. Если engine отсутствует — предложит скачать с ModelScope (~120 МБ)
4. Установит `bin/<os>/config.json` — auth-конфиг движка с общим тестовым API-ключом
5. После скачивания — проверит SHA-256 через `verify.json`
6. Покажет список моделей, отсортированный по совместимости с вашей GPU
7. Для выбранной модели — покажет отсутствующие файлы с кнопкой Download

## QuantFunc API-ключ (важно)

Модели Klein (4B/9B) — **key-protected**: движок при загрузке проверяет лицензию
(`checkObfuscation`). Ключ нужен для расшифровки key map модели. WebUI решает это так:

1. **Тестовый ключ (по умолчанию)** — WebUI сам кладёт `config.json` с общим бесплатным
   тестовым ключом в `bin/<os>/` (тот же механизм, что у ComfyUI-плагина). Действует
   во время public beta, регистрация не нужна.
2. **Свой ключ** — Settings → Engine → «QuantFunc API-ключ». Получить можно на
   https://www.quantfunc.com (кнопка Get API key, бесплатно в beta). Ключ сохраняется
   в settings.json и передаётся движку как `--apikey`, приоритет выше config.json.
3. Если файл конфига удалён/повреждён — кнопка **«Восстановить тестовый ключ»**
   в той же секции (или `POST /api/engine/ensure-config` с `{"force": true}`).

Подробнее: https://www.quantfunc.com/docs/model-loading-and-apikey (раздел 6).

## Использование

1. Выберите модель в правом инспекторе: **Z-Image Turbo** — самая лёгкая (~6.5 ГБ,
   отличный первый тест), **Flux.2 Klein 4B** (~5 ГБ), **Flux.2 Klein 9B** (~10 ГБ)
2. Выберите backend: Lighting (быстрее, нет runtime LoRA) или SVDQ (медленнее, runtime LoRA)
3. Введите prompt в textarea
4. Нажмите **Generate** (или ⌘/Ctrl+Enter)
5. При необходимости — WebUI докачает отсутствующие файлы (с прогрессом)
6. Дождитесь завершения inference (progress bar)
7. Результат отобразится в preview area

## Архитектура

```
Custom HTML/CSS/JS  →  FastAPI + uvicorn  →  Python backend
                                              ↓
                                       Subprocess Worker
                                              ↓
                                       QuantFunc Engine (libquantfunc)
                                              ↓
                                       CUDA  →  NVIDIA GPU
```

Полная архитектурная документация: см. технический отчёт (PDF/DOCX).

## Структура проекта

```
QuantFunc-WebUI/
├── app.py                          # Gradio entry point
├── requirements.txt
├── start.bat                       # Windows launcher
├── settings.json                   # Created on first run (user config)
├── backend/
│   ├── settings.py                 # Pydantic settings store
│   ├── gpu.py                      # GPU detection (nvidia-smi → SM capability)
│   ├── security.py                 # Path + prompt validation
│   ├── cache.py                    # Cache manager (state machine)
│   ├── engine.py                   # Engine binary download + SHA-256 verify
│   ├── worker.py                   # Subprocess wrapper for quantfunc CLI
│   ├── sources/
│   │   ├── base.py                 # ModelSource abstract
│   │   ├── huggingface.py          # HF implementation
│   │   └── modelscope.py           # ModelScope implementation
│   └── registry/
│       ├── loader.py               # ModelRegistry (loads JSON files)
│       └── models/
│           ├── flux_klein_9b.json  # Klein 9B (key_protected)
│           ├── flux_klein_4b.json  # Klein 4B (key_protected)
│           └── z_image_turbo.json  # Z-Image Turbo (без ключа, лёгкая)
├── frontend/
│   ├── css/styles.css              # Dark IDE-grid theme
│   ├── js/
│   │   ├── app.js                  # Bridge JS ↔ Python
│   │   ├── i18n.js                 # Bilingual RU/EN
│   │   └── dropzone.js             # Drag&drop + paste
│   └── locale/
│       ├── en.json
│       └── ru.json
├── bin/<os>/config.json            # Engine auth (server_url + api_key), создаётся автоматически
├── models/cache/                   # Default cache dir (configurable)
└── outputs/                        # Default output dir
```

## Лицензия

- **Код WebUI**: MIT
- **QuantFunc Engine**: Proprietary (распространяется через ModelScope, бесплатно в beta)
- **Model weights**: Apache 2.0 (upstream) + QuantFunc Model License (derivative)

WebUI НЕ включает engine binary или model weights в репозиторий — всё скачивается
автоматически при первом использовании.

## Поддерживаемые модели (Phase 2 + future)

| Модель | Phase | Status |
|--------|-------|--------|
| Flux.2 Klein 9B | Phase 2 | ✅ Implemented |
| Flux.2 Klein 4B | Phase 4 | Pending |
| Qwen-Image | Phase 4 | Pending |
| Qwen-Image-Edit | Phase 6 | Pending |
| Z-Image (Turbo) | Phase 4 | Pending |
| Ideogram-4 | Phase 6 | Pending |
| Qwen-Image-Layered | Phase 6 | Pending |

## Устранение неполадок

| Проблема | Решение |
|----------|---------|
| `nvidia-smi not found` | Установите NVIDIA Driver ≥ 560 |
| `CUDA error: no kernel image` | Драйвер слишком старый для вашей GPU. Обновите драйвер. |
| `cuDNN BAD_PARAM` | Удалите cuDNN algo cache: `rm -rf ~/.cache/cudnn` (Linux) или `%LOCALAPPDATA%\cudnn` (Windows) |
| Engine не скачивается | Проверьте интернет. В Settings переключите source priority на HF. |
| `modelscope not installed` | `pip install modelscope` (нужен для engine binary download) |
| Noisy output | Проверьте, что backend соответствует transformer (svdq vs lighting) |

## Сообщество

- QuantFunc Discord: https://discord.gg/jCp9TpFWcn
- Issues: создайте issue в этом репозитории

---

**Phase 2 status**: ✅ Working text-to-image for Flux.2 Klein 9B
**Phase 3 status**: 🔨 Custom HTML/CSS/JS UI (in progress)
