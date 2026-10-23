# CHANGELOG — LazySD-UI / QuantFunc-WebUI

Журнал коммитов для людей и ИИ-агентов. Перед началом работы сверяйся с
`git log --oneline -20`, потому что этот файл обновляется не автоматически.

> **Последний код-коммит:** `73b9ce8` — fix(ui): keep Dock stationary and repair Models Hub filters (2026-10-23).
> Документация (этот файл и ссылки в MD) идёт отдельным коммитом `docs: ...`, его хеш смотри в `git log -1`.

---

## 2026-10-23

### `73b9ce8` — fix(ui): keep Dock stationary, repair Models Hub filters, tolerate transient 502

Исправлены регрессии, найденные вживую на `127.0.0.1:7871`: сдвиг Dock при
открытии выбора модели в Drag-режиме, неработающие теги/поиск Models Hub и
ошибки `Bad Gateway` в History/Halftone.

Dock:
- Причина сдвига: поповеры лежали в normal flow перед `.gemini-dock`; в
  drag-режиме обёртка привязана к `top`, поэтому её рост увеличивал `top` самой
  панели. Например, при `dockCustomY=120` панель уезжала на `top=378.5`.
- `.dock-popover` теперь `position: absolute; bottom: calc(100% + 8px); left: 0`
  и не влияет на геометрию обёртки. Для случая, когда сверху нет места,
  `toggleDockPopover()` добавляет `.dock-popover-below`
  (`top: calc(100% + 8px)`).
- Проверено в браузере: Dock остаётся на `top=120` при открытии поповера и
  вверху, и внизу экрана; смещение — `0 px`. Ручка Drag и inline `left/top`
  не меняются.

Models Hub:
- Исправлен селектор активной кнопки: реальная разметка —
  `#models-tab-filters .filter-chip` с `data-tab-filter`, а не
  `#models-hub-filters .filter-pill`.
- `renderModelsTab()` реально фильтрует `all`, `installed`, `custom`, `hidden`,
  `svdq`, `sd_cpp`, `sdnq`, `klein`, `qwen`, `z-image`; поиск идёт по id,
  названию, family, engine, описанию, текстовым энкодерам и вариантам.
- `hidden` учитывает оба флага: `hidden_from_host` и `hidden_for_guests`.
- Ветка `encoders` запрашивает `/api/encoders` и рисует отдельные карточки
  (`renderEncodersTab`) с кнопкой скачивания; после скачивания сбрасывается
  кэш энкодеров и вызывается `refreshConfig()`.
- `frontend/js/modules/models-hub.js` синхронизирован: тот же селектор,
  `hidden`-фильтр и обновление active по `data-tab-filter`.

History/Halftone и сетевые ошибки:
- Живые запросы `/api/history?page=1..3` и `/api/halftone/process` на CPU
  вернули HTTP 200; в `logs/verify/quantfunc.log` не было ни 502, ни traceback.
  Ошибки `Bad Gateway` возникали, когда проверочный сервер был недоступен:
  запросы вообще не доходили до uvicorn.
- Добавлен `_fetchWithRetry`: GET-запросы повторяются до 2 раз, превью
  Halftone — 1 раз, только для `502/503/504` и сетевых ошибок. Мутирующие
  POST-генерации не повторяются, чтобы не создать дубликаты задач.
- Проверено инъекцией ответов: два подряд 502 на History и один 502 на
  Halftone preview успешно перекрываются повтором, UI не показывает ошибку.

Проверка:
- `python -m pytest tests -q` → **36 passed**; статические guard-тесты
  дополнены проверками Dock-поповеров, фильтров и retry.
- Все 14 frontend JS-файлов прошли `node --check`; `compileall` и оба locale
  JSON валидны.
- Живые фильтры: 23 all, 7 installed, 0 custom, 1 hidden, 13 encoders, 7 SVDQ,
  6 SD.CPP, 1 SDNQ, 11 Klein, 7 Qwen, 2 Z-Image; поиск `klein` — 10 карточек.
- История: страницы 1–3 по 24 элемента, без тостов; Halftone CPU: 1024×1024,
  1536×1536 и 2048×2048, HTTP 200, без `Halftone error`.
- Bump cache-bust: `/js/app.js?v=20261023_04`, `/css/styles.css?v=20261023_04`.

### `d322737` — feat(queue): serialize GPU jobs, add presence monitor, fix History and Dock drag

Основная задача: не допускать параллельного запуска GPU-генерации из нескольких
вкладок/аккаунтов, показать понятную очередь, добавить мониторинг подключений и
починить зависания History, а также конфликт Drag-режима Gemini Dock.

GPU-очередь:
- Новый `backend/generation_queue.py`: FIFO-очередь с одним активным job и
  поддержкой отмены ожидающих/активных задач.
- `AppState` хранит `owner_id`, `owner_name`, `kind`, `shared`, выдаёт
  `queue_position`, `queue_size`, `queue_active`; активная задача получает слот
  только после `acquire_generation()`, освобождает в `finally`.
- Сериализованы ровно две GPU-точки: `POST /api/generate` и
  `POST /api/manga/ai-job`. Скачивания, Halftone и Manga CPU-обработка остаются
  параллельными.
- Отмена ожидающей задачи удаляет её из очереди и не трогает чужой активный
  воркер; отмена активной останавливает worker. Поздняя ошибка воркера не
  перезаписывает статус `cancelled`. Если cancel пришёл между активацией слота
  и записью `initializing`, слот освобождается, а job остаётся `cancelled`.
- Для Halftone/Manga CPU-пайплайнов добавлен общий семафор
  `TOOL_CPU_MAX_CONCURRENCY = 4` (3–4 одновременные задачи).
- Глобальный `#queue-overlay` показывает позицию, число ожидающих и кнопку
  отмены на всех вкладках; исчезает, как только job переходит из `queued` в
  `initializing/loading/generating`.

Подключения:
- Новый `backend/presence.py`: уникальные аккаунты по `role:id`, TTL 90 секунд.
- `ShareAccessMiddleware` отмечает активность, фронтенд шлёт heartbeat каждые
  20 секунд; добавлены `/api/presence` и `/api/presence/heartbeat` (гостям
  разрешены, но список пользователей видит только хост).
- Настройка `monitor_connections` (по умолчанию включена) в Settings, PATCH и
  `/api/config`; в шапке появился постоянный индикатор `#presence-indicator`.
- В настройках хоста — счётчик онлайн и список подключённых пользователей.

History:
- Новый `backend/history_paging.py`; `/api/history` принимает `page`,
  `page_size`, `filter`, `q` и возвращает
  `{items,total,page,page_size,pages}`. Размер страницы 12/24/48, серверный
  максимум 100, номер страницы клампится к последней.
- Без query-параметров сохранён старый ответ-массив для совместимости.
- `CacheManager.list_outputs(limit=None)` отдаёт все файлы для пагинации.
- Фронтенд: серверные фильтры, debounce поиска, выбор размера страницы,
  кнопки страниц, защита от гонки устаревших ответов (`historyRequestSeq`) и
  снятие `history-loading` в `finally` — зависаний при ошибке больше нет.

Dock:
- Drag вынесен в отдельный явный режим (`#dock-drag-toggle-btn` и кнопка в
  настройках). Стартует только с ручки `⋮⋮`, а не со всей коробки/поповеров.
- Убран `left: auto !important` из `.gemini-dock-wrap.dock-free-drag`, который
  перебивал inline-координаты и вызывал движение только в одну сторону.
- Оба axes клампятся в границы окна; смена preset-позиции выключает drag;
  координаты и флаг сохраняются в `localStorage["lazysd_dock"]`.
- Добавлены сброс (`#dock-reset-btn`, двойной клик по ручке) и восстановление
  прежней свободной позиции при повторном включении Drag.

Проверка:
- `pytest tests -q` → **36 passed** (добавлен
  `tests/test_queue_presence_history.py`).
- Все 14 JS-файлов прошли `node --check`; оба locale JSON валидны;
  `python -m py_compile app.py backend/*.py` проходит.
- Живой запуск на `127.0.0.1:7871`: два FLUX.3-задания 512×512/4 шага —
  A `loading`, B `queued position=1`; отмена B не затронула A; A завершился за
  34.0 с, слот освобождён.
- В браузере: History 249 работ / 11 страниц, переход на страницу 2, размер
  страницы 12, поиск по prompt; очередь-оверлей, presence-индикатор и Dock
  drag/clamp/reset проверены через реальный UI.
- Bump cache-bust: `/js/app.js?v=20261023_03`, `/css/styles.css?v=20261023_03`.

### `0470af4` — fix(flux3): load Nunchaku FLUX.3 via diffusers ModularPipeline

Причина ошибки `[generate] Job cd33523f failed`:
`NunchakuFlux3Transformer2DModel requires diffusers with Flux3Transformer2DModel`.
В `envs/extras` стоял стабильный `diffusers 0.41.0` с PyPI: в нём есть
`Flux2KleinPipeline`, но нет `Flux3Transformer2DModel` и `ModularPipeline`.
FLUX.3 появился только в GitHub main (`0.42.0.dev0`, проверено на коммите
`48cdbaef5ee3d718aaee392b44f2099ae511c4a5`). Установщик Extras проверял только
`has_flux2_klein`, поэтому не обновлял diffusers для FLUX.3.

Исправлено:
- `backend/extras.py`: проба окружения теперь сообщает `has_diffusers_flux3` и
  `has_modular_pipeline`; установщик ставит diffusers из GitHub main, если этих
  классов нет.
- FLUX.3 Nunchaku загружается через `ModularPipeline` / `Flux3DistilledModularPipeline`,
  а не через `Flux2KleinPipeline`.
- Локальный `modular_model_index.json` распознаётся как индекс пайплайна.
- INT4-энкодер FLUX.3 загружается классом `NunchakuQwen3VLEncoderModel`
  (`Qwen3-VL-4B`), а не Qwen2-VL/Qwen3-text.
- Блок `prompt_upsampler` отключается: отдельная 9B-модель апсемплера пока не
  управляется WebUI, поэтому промпт идёт в текстовый энкодер как есть.
- Обработка результата ModularPipeline: список изображений / `output="images"`,
  без неподдерживаемых `guidance_scale` и `callback_on_step_end`.
- `frontend/js/app.js` и `modules/generator.js`: перед генерацией FLUX.3
  проверяют не только Nunchaku >= 1.4.0, но и наличие diffusers main; показывают
  понятное сообщение о переустановке Extras.
- Добавлен тест `test_flux3_nunchaku_support`.
- Bump cache-bust: `/js/app.js?v=20261023_02`.

Проверка:
- `pytest tests -q` → **29 passed**.
- Все JS-файлы (9) прошли `node --check`.
- Реальные задания через WebUI/backend: 512×512, 4 шага — 25.8 с;
  1024×1024, 4 шага — 45.6 с. Оба изображения сохранены в `outputs/`.

### `2e1b352` — fix(lora): restore LoRA Hub rendering and local library detection

Причина: активный монолит `frontend/js/app.js` рассинхронизировался с
`frontend/index.html` и `frontend/js/modules/lora-hub.js`.

Исправлено:
- `switchModelsHubSubtab('loras')` теперь реально показывает `#models-subpanel-loras`
  и скрывает панель моделей.
- Добавлены рабочие `renderLoraHubGrid()` и `renderLoraLibrary()`; LoRA Hub
  показывает локальные файлы, счётчик, фильтр, триггеры и кнопки подключения.
- Исправлены вкладки модального окна (`hub` / `local` / `library`).
- HF-запросы переведены на реальные эндпоинты `/api/loras/hf-inspect` и
  `/api/loras/hf-search`; скачивание читает `#lora-hf-file-select`.
- `app.py`: открытие папки `loras` ведёт в `models_dir/loras`
  (фактически `models/cache/loras`).
- `modules/lora-hub.js` синхронизирован с активной версией.
- Добавлен регрессионный тест `test_lora_hub_frontend_sync`.
- Bump cache-bust: `/js/app.js?v=20261023_01`.

Проверка: `pytest tests -q` → **28 passed**; все JS прошли `node --check`;
в браузере LoRA Hub показывает 1 локальную LoRA
(`[FLUX.2.Klein]LineArt_Redmond.safetensors`).

### `9ece51d` — feat(halftone): add layer stack, masks, and WebGL2 FBO compositor

- Система слоёв Halftone Studio GT: заблокированный виртуальный оригинал,
  автоматический базовый растр, локальные слои с пустыми масками.
- Многопроходный WebGL2 FBO-композитор с масками, opacity и blend modes.
- Перемещение слоёв ⬆/⬇, дублирование, удаление, бейджи base/mask.
- Защита от муара для `Multiply` (сдвиг угла на 15°).
- Бэкенд пропускает виртуальный оригинал и сохраняет `hasMask`/`isBase`.
- Тест `test_halftone_layer_stack_system`; на тот момент **27 passed**.

---

## 2026-10-17

| Коммит | Что изменилось |
|---|---|
| `dcc5891` | Кнопка сохранения пресета в доке и модальном окне; исправлен probe Nunchaku; HF-токен синхронизирован с CLI env. |
| `27bb549` | Безопасность удалённого доступа: экран входа, удалён backdoor `guest_public`, чистые share-URL. |
| `7d3b401` | Phase 1–2: аккаунты, compare curtain, пресеты, видимость моделей, HF Hub и пользовательские модели. |
| `9ca1294` | Компактная статистика CLI-движка, настройка HuggingFace-токена, изоляция гостей и шаринг. |
| `2bb835e` | Автоустановка cloudflared, исправление payload установки Extras и пояснения в UI. |
| `e174f76` | Возврат открытия локальных папок outputs/models в Проводнике Windows. |

## 2026-10-16

| Коммит | Что изменилось |
|---|---|
| `da394dd` | Compare curtain, multi-LoRA stacking, стили/докинг дока, CPU/GPU-статистика, OrbitQuant & Nunchaku, шаринг без Gradio. |
| `b658bae` | Плавающий Gemini Dock, Topaz-инспектор, 3-уровневая сортировка моделей, HF explorer, Presets hub. |
| `8784641` | Нативный ControlNet/Reference для Flux и Qwen, live VRAM/RAM, исправления history modal и prompt resize, RTX 5070 Ti cudart. |
| `aa4980e` | Переименование в LazySD UI, FIT preview, раскладка настроек, сворачиваемый sidebar, отключение QuantFunc-моделей по умолчанию. |

---

## Правила ведения журнала

1. Каждый значимый коммит добавляй в этот файл отдельной записью: хеш, тип,
   дата, что изменено и как проверено.
2. Если коммит меняет поведение Halftone/LoRA/Manga, обнови соответствующий
   раздел `AGENT_HANDOFF.md`.
3. Для активного фронтенда источник истины — `frontend/js/app.js`.
   `frontend/js/modules/*` обязан оставаться синхронизированным.
4. Перед коммитом запускай:
   ```powershell
   node --check frontend/js/app.js
   python -m pytest tests -q
   ```
