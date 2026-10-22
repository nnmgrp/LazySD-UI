# CHANGELOG — LazySD-UI / QuantFunc-WebUI

Журнал коммитов для людей и ИИ-агентов. Перед началом работы сверяйся с
`git log --oneline -20`, потому что этот файл обновляется не автоматически.

> **Последний код-коммит:** `0470af4` — fix(flux3): load Nunchaku FLUX.3 via diffusers ModularPipeline (2026-10-23).
> Документация (этот файл и ссылки в MD) идёт отдельным коммитом `docs: ...`, его хеш смотри в `git log -1`.

---

## 2026-10-23

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
