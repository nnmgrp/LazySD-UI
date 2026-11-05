# Отчёт об исследовании: артефакты чернил и мазков кисти при Inpaint

* **Дата:** 03 ноября 2026 г.
* **Объект:** `QuantFunc-WebUI` / `LazySD-UI`, ветка `1048d2d`
* **Исследуемый инцидент:** генерация `outputs/nunchaku_20261103_031830_1793657910.png` (seed `1793657910`, FLUX.2 Klein 9B Nunchaku SVDQ)
* **Исходный документ:** `docs/INPAINT_INK_BRUSH_SPEC.md`

---

## 0. Итог в одном абзаце

Диагноз из ТЗ **подтверждён полностью и подтверждён на уровне пикселей**: артефакты вызваны исключительно текстом промпта, а не «призраками» VAE и не настройками маски. Но ТЗ **неполно**: оно закрывает 2 из 9 мест, где в код зашита лексика «кисть/тушь», оставляет нетронутым few-shot пример, который и породил основную часть вредных слов, вводит регексы с доказанными ложными срабатываниями, и не покрывает два пути исполнения, где любой фикс вообще не применится. Ниже — доказательная база, полный перечень точек и скорректированный план.

---

## 1. Доказательная база

### 1.1. Отпечаток negative_prompt однозначно определяет путь генерации

Манифест генерации содержит:

```json
"negative_prompt": "text, japanese, kanji, kana, letters, sfx, speech bubble outline, watermark, signature, seams, blurry, low contrast, color shift, japanese sfx, katakana, speech bubble, visible seam"
```

Прогон через реальный код проекта (`_sanitize_negative_prompt(DEFAULT_NEGATIVE_PROMPT, instruction="delete this Japanese SFX")`) даёт **побайтово идентичную строку**:

```
MATCH : True
```

Это возможно только при выполнении двух условий одновременно:

1. Ветка `parse_vlm_response_text` отдала **prose**, а не JSON. JSON-ветка (`backend/vision_runtime.py:334`) вернула бы текст из ответа модели, а не `DEFAULT_NEGATIVE_PROMPT`. Prose-ветка (`backend/vision_runtime.py:348-352`) возвращает ровно `DEFAULT_NEGATIVE_PROMPT` — без санитизации и без обязательных токенов.
2. Затем `generate_vlm_prompt` дописал обязательные токены удаления (`japanese sfx`, `katakana`, `speech bubble`, `visible seam`) — механизм `backend/vision_runtime.py:122-126`.

Отсюда: **использован путь FLUX 3 PUP (Nunchaku), режим prose**. Это подтверждается и тем, что `models/cache/flux_3_pup_nunchaku` была установлена 03.11.2026 в 00:38 — за 2,5 часа до инцидента, тогда как Ollama сейчас недоступна, а `settings.py:159-160` по умолчанию указывают на `ollama` / `qwen2.5vl:3b`.

### 1.2. Pre-heal работал идеально — «призраки» исключены

`heal_masked_image` (`backend/masking.py:714`) заполнил область маски до диффузии. Замеры по `outputs/masks/nunchaku_init_1793657910_28788.png` и `outputs/masks/nunchaku_mask_28788.png`:

| Область | mean RGB | std | тёмных пикселей |
|---|---|---|---|
| Ядро маски (`mask > 0.9`, n = 218 314) | **79.6 / 54.1 / 75.2** | **6.4 / 4.9 / 4.4** | **0.00 %** |
| Мягкий край (`0.03 < mask ≤ 0.9`) | 70.2 / 45.6 / 68.0 | 11.9 / 7.5 / 6.8 | 0.00 % |
| Вне маски | 79.8 / 54.7 / 75.2 | 84.0 / 85.2 / 77.0 | 2.80 % |

Внутри маски — идеально ровная заливка с отклонением ±5 на канал и **нолём чёрных пикселей**. VAE физически не могла «вытащить» контур SFX: его там не было.

> **Поправка к ТЗ.** В §2.4 ТЗ сказано «чёрных пикселей 0,0,0 не было». Это верно **для ядра маски**, но не для всего файла: во всём init-изображении 6 758 чисто чёрных пикселей — это исходные чёрные мазки SFX **вне** маски. Формулировку стоит уточнить, иначе при проверке ТЗ она даст ложный провал.

### 1.3. Диффузия исполнила промпт буквально

Замеры финального PNG в тех же координатах (`mask_crop.box = [176, 2992, 912, 3696]`):

| Область | mean RGB | std | тёмных пикселей |
|---|---|---|---|
| **Внутри маски** | **88.4 / 14.6 / 29.5** | **45.2 / 9.8 / 13.7** | **17.10 %** |
| Вне маски | 78.3 / 53.4 / 74.1 | 85.5 / 86.3 / 77.8 | 37.10 % |

Ровная сливово-сиреневая заливка `[80, 54, 75]` заменена на багровую `[88, 15, 30]` с чёрными мазками. Сопоставление с промптом:

| Слово промпта | Что появилось в пикселях |
|---|---|
| `dark red` / `crimson washes` | среднее ушло в R 88 → G 15 (багровый) |
| `black brush strokes`, `dry-brush streaks` | чёрные штрихи, 17 % тёмных пикселей |
| `scattered droplets`, `rough manga-style splatter` | круглые брызги и капли |

![Доказательная фигура](../outputs/_investigation/ink_brush_evidence.png)

### 1.4. Главный виновник — few-shot пример в system prompt

Сгенерированный промпт лексически почти цитирует пример из `INPAINT_PUP_SYSTEM_PROMPT` (`backend/vision_runtime.py:366-368`):

> **Пример в system prompt:** `"Dark crimson and deep burgundy manga background with rough vertical ink brushwork, subtle plum shadows, painterly anime illustration texture, clean seamless surface"`
> **Сгенерированный промпт:** `"A seamless dark red and black ink-splash background filling the entire area, with dense crimson washes, irregular black brush strokes, dry-brush streaks, scattered droplets, and rough manga-style splatter matching the surrounding artwork."`

Совпадение по `crimson`, `ink`, `manga`, `brush`, `seamless`, `texture` — практически дословное. **Решающий фактор — не инструкция «опиши текстуру кисти манги», а конкретный пример с чернильными мазками, который модель копирует.** Инструкции против примеров в LLM работают по правилу «пример сильнее запрета».

---

## 2. Полный перечень точек с лексикой «кисть / тушь / текстура»

ТЗ закрывает **2 из 9**. Ниже — реестр по всему коду.

| # | Файл:строка | Фрагмент | Закрыто ТЗ? |
|---|---|---|---|
| 1 | `vision_runtime.py:52` (`SYSTEM_PROMPT`, общий Ollama-путь) | `"…and brush texture that should seamlessly fill the red shapes"` | ❌ нет |
| 2 | `vision_runtime.py:361-362` (`INPAINT_PUP_SYSTEM_PROMPT`, правило 1) | `"…brushwork texture that should appear inside that area"` | ⚠️ частично (§4.1.1) |
| 3 | **`vision_runtime.py:366-368`** (тот же system prompt, правило 4 — **few-shot пример**) | `"…rough vertical **ink brushwork**… painterly anime illustration texture"` | ❌ **нет** — самый сильный драйвер |
| 4 | `vision_runtime.py:550` (Nunchaku PUP, removal) | `"…lighting, and **manga/anime brush texture** that should fill this area"` | ✅ §4.1.2 |
| 5 | `vision_runtime.py:562` (Nunchaku PUP, **пустая инструкция**) | `"…and **brush texture** to restore this area"` | ❌ нет |
| 6 | `vision_runtime.py:589` (FLUX 3 PUP через Ollama, removal) | `"…lighting, and **manga/anime brush texture** that should fill this area"` | ❌ нет (ТЗ цитирует только Nunchaku-версию, стр. 547-552) |
| 7 | `vision_runtime.py:601` (FLUX 3 PUP через Ollama, **пустая инструкция**) | `"…and **brush texture** to restore this area"` | ❌ нет |
| 8 | `vision_runtime.py:643` (общий Ollama-путь, removal) | `"…surface, lighting, colors, and **art texture** that should fill the red masked shapes"` | ❌ нет (мягче, но логика та же) |
| 9 | `prompt_fallback.py:376` (эвристика, low-saturation) | `"manga screentone and **ink** lineart texture"` | ❌ нет |
| 10 | `prompt_fallback.py:378` (эвристика, saturated) | `"manga **brush** and illustration texture"` | ✅ §4.4 |
| 11 | `prompt_fallback.py:18-21` (`DEFAULT_NEGATIVE_PROMPT`) | без стоп-токенов `ink`/`brush` | ❌ нет |
| 12 | `frontend/js/app.js:13060-13078` (пресеты Inpaint) | `clean_text` — без ink/brush; `lineart` — `"black ink line drawing"` | ❌ нет |

Точки 1, 3, 5, 6, 7 — это **пять дополнительных** мест, где модель получает прямое указание описывать мазки, плюс два «пустая инструкция»-сценария (пользователь просто замазал SFX и нажал «Сгенерировать промпт» без текста — самый частый сценарий).

---

## 3. Риски и пробелы в текущем ТЗ

### G1. 🔴 Критично — few-shot пример остаётся в system prompt

§4.1.1 предлагает **добавить** блок «STRICTLY FORBIDDEN WORDS: brush, stroke, ink, splatter…», но не предлагает **удалить** правило 1 (`brushwork texture`) и правило 4 с примером (`rough vertical ink brushwork`). Реализация ТЗ «как есть» даст системный промпт, который одновременно запрещает `ink`/`brush` **и** показывает пример с `ink brushwork`. Противоречие разрешится в пользу примера — то есть ровно в пользу текущего дефекта.

**Вывод:** §4.1.1 требует не «дополнения», а **полной замены** `INPAINT_PUP_SYSTEM_PROMPT`, где пример заменяется на нейтральный (например, `"flat uniform mauve background, smooth solid fill, even lighting, seamless surface"`).

### G2. 🔴 Регексы ТЗ ломают легитимные промпты

Предложенный список `_INPAINT_ARTEFACT_PATTERNS` применён безусловно (в §4.2 флаг `is_removal` упомянут только в сигнатуре). Проверка на реальных формулировках:

| Исходный промпт | После санитизации по ТЗ | Оценка |
|---|---|---|
| `neon light streaks across a rainy city skyline` | `neon light across a rainy city skyline` | ❌ сломано |
| `water splashes on the shore, golden sunset` | `water on the shore, golden sunset` | ❌ сломано |
| `ink wash painting of a mountain landscape` | `painting of a mountain landscape` | ❌ сломано |

**Вывод:** паттерны обязаны применяться **только при `is_removal=True`**. Иначе фикс чинит SFX-удаление, но ломает обычную генерацию. ТЗ этого не требует явно.

### G3. 🔴 Два пути исполнения обходят любой санитизатор

`parse_vlm_response_text` (`backend/vision_runtime.py:312`) имеет две ветки:

* **JSON** (строка 334) — санитизирует промпт и negative сам. ✅
* **Prose** (строки 348-352) — возвращает `combined[:1500]` **без `_sanitize_positive_prompt`** и `DEFAULT_NEGATIVE_PROMPT` **без обязательных токенов**. ❌

Дальше:
* `generate_vlm_prompt` → Nunchaku-ветка (строка 574-575) **перевызывает** оба санитизатора → prose чинится. ✅
* `generate_vlm_prompt` → Ollama-ветка (строка 683) `return parse_vlm_response_text(...)` **возвращает напрямую** → prose **не чинится вообще**. ❌

При этом Ollama-ветка FLUX 3 PUP (строки 603-624) **не задаёт `format: "json"`** (в отличие от общей Ollama-ветки, строка 658) — то есть эта модель **всегда** отвечает prose и **всегда** обходит санитизатор.

**Вывод:** изменения §4.2/§4.3 не дойдут до Ollama-FLUX-3-PUP-пути. Требуется либо санитизация внутри prose-ветки `parse_vlm_response_text`, либо явный вызов санитизаторов на строке 683.

### G4. 🟠 Кэш промптов никогда не инвалидируется

* `_PROMPT_CACHE` (`backend/prompt_gen.py:52-53`), лимит 50 записей.
* VLM-результаты кэшируются (`prompt_gen.py:434-435`) — то есть **плохой промпт застревает**.
* `_compute_cache_key` (`prompt_gen.py:220-248`) включает путь/mtime/хэш маски/инструкцию/модель/crop — **но не версию шаблонов промпта**.
* `clear_prompt_cache()` (`prompt_gen.py:60-62`) **не вызывается нигде в проекте** (проверено grep по `backend/*.py` и `app.py`).

**Вывод:** после исправления шаблонов пользователь продолжит получать старый плохой промпт для того же (изображение, маска, инструкция, модель) до перезапуска процесса. Нужна «соль» версии шаблона в ключ кэша (или вызов `clear_prompt_cache()` при старте/смене настроек).

### G5. 🟠 Эвристика не различает задачу удаления

`prompt_fallback.py:375-379` выбирает текстуру по `region_type` (насыщенность), **а не по типу задачи**. Для задачи удаления SFX на насыщенной сцене эвристика выдаст `"manga brush and illustration texture"` — тот же класс дефекта. §4.4 меняет формулировку, но не делает эвристику *removal-aware*. Также пропущена ветка `prompt_fallback.py:376` (`"manga screentone and ink lineart texture"`).

### G6. 🟡 `DEFAULT_NEGATIVE_PROMPT` в бэкенде не обновляется

`prompt_fallback.py:18-21` — это база negative для эвристики **и** для prose-ветки `parse_vlm_response_text`. §4.5.1 меняет только фронтенд-поле `inpaint-negative-prompt`, а в `index.html:1469` у textarea **вообще нет** дефолтного `value` — значение приходит из `localStorage` или перезаписывается ответом ассистента. То есть фронтовое «дефолтное значение» из ТЗ технически некуда положить; реальный рычаг — бэкендовый `DEFAULT_NEGATIVE_PROMPT` + `_MANDATORY_REMOVAL_NEGATIVES`.

### G7. 🟡 Чекбокс «Плоская заливка» не имеет контракта API

Модель запроса `prompt_gen.py:260-266` (`source_path`, `mask`, `instruction`, `crop_*`, `vision_model`, `vision_provider`, `use_vlm`) не содержит поля для этого флага. Фронт отправляет запрос на `app.js:14501-14511`. Нужно завести поле (например `flat_fill: bool`) и протянуть его `app.js → prompt_gen → generate_vlm_prompt → выбор шаблона`.

### G8. 🟡 Дубликат фронтенд-кода (ловушка расхождения)

`frontend/js/modules/inpaint.js` содержит **копию** всей логики ассистента (`/api/inpaint/prompt` на строке 2065, выбор модели на 1893-2091). Он подключается через `frontend/js/main.js:20`, но `main.js` **не загружается** в `frontend/index.html` (в `index.html` подключены только `i18n.js`, `dropzone.js`, `app.js`). То есть `modules/inpaint.js` — мёртвый код. Любой фронтовый фикс по ТЗ §4.5 надо вносить в `app.js`; расхождение копий — ловушка для будущих правок.

---

## 4. Скорректированный план реализации

### Этап 1 — убрать источник (backend, `vision_runtime.py`)

1. **Полностью переписать `INPAINT_PUP_SYSTEM_PROMPT` (строка 357-369):**
   * убрать `brushwork texture` из правила 1 → `"…of the clean surface, background, lighting and colors"`;
   * **заменить** few-shot пример (правило 4) на нейтральный: `"flat uniform mauve background, smooth solid fill, even lighting, seamless surface"`;
   * добавить блок «STRICTLY FORBIDDEN WORDS» из §4.1.1 ТЗ.

2. **Переписать `SYSTEM_PROMPT` (строка 43-58)** — правило 3: убрать `and brush texture`.

3. **Исправить оба removal-шаблона** (`строки 546-552` и `585-591`) по §4.1.2 ТЗ — заодно **вынести в общий хелпер**, чтобы Nunchaku и Ollama-ветки не расходились (сейчас это два дубликата).

4. **Исправить оба шаблона «пустой инструкции»** (`строки 559-563` и `598-602`): `brush texture to restore this area` → `flat smooth solid fill matching surrounding tones`.

5. **Ослабить строку 643:** `art texture` → `flat solid color fill`.

### Этап 2 — фильтр-страховка (backend)

6. **`_sanitize_positive_prompt(prompt_raw, is_removal=False)`** — добавить флаг; **применять артефакт-паттерны только при `is_removal=True`** (устранение G2). Дополнить список ТЗ: добавить `texture`, `brushwork`, `hatching`, `crosshatch`, `screentone`(?), а также `crimson`/`scarlet`/`blood-red` в сочетании с `wash`/`pool`/`stain`.

7. **Протянуть `is_removal` по всем вызовам** (`vision_runtime.py:334`, `575`, `683`) — и **добавить санитизацию в prose-ветку** `parse_vlm_response_text` (строки 348-352), чтобы Ollama-путь тоже чинился (устранение G3).

8. **`_sanitize_negative_prompt`:** расширить `_MANDATORY_REMOVAL_NEGATIVES` списком из §4.3 ТЗ. Учесть, что фильтр `^(?:no|without|not)\s+` превращает `no ink` → `ink` — это нормально, но `"black marks"` и `"dark spots"` не должны дублировать уже присутствующие токены (дедуп по `seen` уже работает).

9. **Обновить `DEFAULT_NEGATIVE_PROMPT` (`prompt_fallback.py:18-21`)** — добавить `ink, brush strokes, dry brush, splatter, streaks, dark spots` (устранение G6).

### Этап 3 — эвристика и кэш

10. **`prompt_fallback.py:375-379`:** сделать выбор текстуры зависимым от `is_removal` — для задач удаления всегда `flat smooth clean background surface, uniform color fill, seamless solid backdrop` (как в §4.4 ТЗ) + `no ink, no brush marks`. Не забыть ветку `screentone and ink lineart` (строка 376).

11. **Кэш:** добавить в `_compute_cache_key` константу `PROMPT_TEMPLATE_VERSION` (инкремент при изменении шаблонов) **или** вызвать `clear_prompt_cache()` на старте сервиса (устранение G4).

### Этап 4 — интерфейс

12. `app.js`: обновить/добавить пресеты `clean_text` (negative с ink/brush) — `app.js:13060-13078`.
13. `app.js:14501-14511`: добавить поле `flat_fill` в запрос; добавить чекбокс «Плоская заливка (без мазков и фактуры)» в блок ассистента, **активный по умолчанию**; чекбокс должен влиять на шаблон, а не только на текст подсказки.
14. `index.html:1469`: задать дефолт `value` у `inpaint-negative-prompt` (чтобы §4.5.1 ТЗ имел куда легнуть) — либо, что чище, перенести дефолт в бэкенд и не полагаться на значение поля.

### Этап 5 — серверный контракт

15. `prompt_gen.py`: добавить `flat_fill: bool = True` в сигнатуру и в `_compute_cache_key`; протянуть в `generate_vlm_prompt`.

---

## 5. Тест-план (сводка)

Текущие тесты: `tests/test_prompt_fallback.py`, `tests/test_prompt_gen_api.py` — санитизаторов промпта не покрывают вовсе.

| ID | Тест | Ожидание |
|---|---|---|
| **T-1** | `_sanitize_positive_prompt` на реальном промпте из манифеста | не содержит `brush`, `ink`, `splatter`, `streak`, `droplet`, `wash` |
| **T-2** | `_sanitize_positive_prompt("neon light streaks...", is_removal=False)` | **строка не изменена** (регрессия G2) |
| **T-3** | `_sanitize_positive_prompt("water splashes...", is_removal=True)` | содержит `splashes` (не removal-контекст верен) |
| **T-4** | `_sanitize_negative_prompt(..., instruction="удалить SFX")` | содержит `ink`, `brush strokes`, `splatter`, `dry brush` |
| **T-5** | `_sanitize_negative_prompt(..., instruction="remove the car")` | **без** `ink`/`brush` — стоп-слова чернил привязаны к удалению текста/SFX, а не к любому удалению |
| **T-6** | `parse_vlm_response_text` на **prose**-ответе | промпт пройден через `_sanitize_positive_prompt`, negative содержит обязательные токены (регрессия G3) |
| **T-7** | `INPAINT_PUP_SYSTEM_PROMPT` / все `req_text` | строка не содержит `brush`, `ink`, `texture` (линтер по шаблонам) |
| **T-8** | `generate_heuristic_prompt` с инструкцией удаления | промпт не содержит `brush`/`ink`/`texture` (регрессия G5) |
| **T-9** | два запроса с одним ключом кэша, но разной `PROMPT_TEMPLATE_VERSION` | ключи различаются (регрессия G4) |
| **T-10** | end-to-end: `POST /api/inpaint/prompt` с `flat_fill=True` | response `prompt` без артефактных слов, `negative_prompt` с `ink`/`brush` |

---

## 6. Уточнённые критерии приёмки

К AC-1…AC-4 из ТЗ добавить:

* **AC-5.** Ни один шаблон промпта в `backend/vision_runtime.py` и `backend/prompt_fallback.py` не содержит слов `brush`, `ink`, `brushwork`, `splatter` (проверяется тестом-линтером T-7, чтобы регрессия не вернулась).
* **AC-6.** Санитизатор **не** изменяет промпты, не относящиеся к удалению (тест T-2).
* **AC-7.** Prose-ответ VLM проходит через санитизатор и получает обязательные negative-токены (тест T-6).
* **AC-8.** После деплоя фикса ранее закэшированный плохой промпт **не** возвращается (тест T-9).
* **AC-9.** Визуально: при удалении SFX область маски сохраняет цвет окружения (≈ `79.6 / 54.1 / 75.2`), а не уходит в багровый (`R` ≥ 80 при `G` < 30); доля тёмных пикселей внутри маски < 1 %.

---

## 7. Что подтверждено, что не подтверждено

**Подтверждено измерениями:**
* Артефакты вызваны текстом промпта, а не VAE-призраками и не настройками маски (§1.2, §1.3).
* Плохой промпт получен через FLUX 3 PUP (Nunchaku) в prose-режиме (§1.1).
* Pre-heal из коммита `1048d2d` отработал корректно (§1.2).
* Регексы из ТЗ §4.2 ломают легитимные промпты при безусловном применении (§3 G2).
* Ollama-FLUX-3-PUP путь обходит санитизаторы при prose-ответе (§3 G3).
* `clear_prompt_cache()` не вызывается нигде (§3 G4).

**Не подтверждено (нет данных в репозитории):**
* Точный VLM, выдавший промпт, — установлен по отпечатку negative_prompt (высокая уверенность), но прямых логов запроса в проекте нет.
* Поведение модели после исправления system prompt нельзя предсказать без прогона — поэтому фильтр-страховка (§4 этап 2) обязателен, а не опционален.
* «Чёрных пикселей 0,0,0 не было» из ТЗ §2.4 — верно только внутри ядра маски, во всём файле их 6 758.

---

*Скрипт построения доказательной фигуры: `outputs/_investigation/ink_brush_evidence.py` (воспроизводим, `python outputs/_investigation/ink_brush_evidence.py`).*

---

## 8. Статус реализации

> **Внимание:** этот раздел описывает первый раунд исправления (03.11.2026, ~04:00).
> После пользовательской проверки выяснилось, что заливка стала плоским красным
> пятном — это второй, более глубокий корневой дефект. См. **§9**.

Реализовано по плану §4. Изменённые файлы:

| Файл | Что сделано |
|---|---|
| `backend/vision_runtime.py` | Переписаны `SYSTEM_PROMPT`, `INPAINT_PUP_SYSTEM_PROMPT` (+ новая removal-версия); все 6 шаблонов `req_text` заменены на общий хелпер `_build_inpaint_request()`; добавлены `_INPAINT_ARTEFACT_PATTERNS`, `_INPAINT_META_PATTERNS`, `_strip_inpaint_artifacts()`; `_sanitize_positive_prompt(prompt, is_removal=False)`; стоп-слова чернил в `_sanitize_negative_prompt()`; prose-ветка `parse_vlm_response_text()` теперь санитизируется |
| `backend/prompt_fallback.py` | Ветка `is_remove` в эвристике (плоская заливка вместо «manga brush … texture»); `MANDATORY_INK_NEGATIVES` как общий источник правды |
| `backend/prompt_gen.py` | `PROMPT_TEMPLATE_VERSION` в ключе кэша; `flat_fill` в контракте и в ключе кэша |
| `app.py` | Поле `flat_fill` в `POST /api/inpaint/prompt` |
| `frontend/js/app.js` | Отправка `flat_fill`; пресет `clean_text` переписан на плоскую заливку |
| `frontend/index.html` | Чекбокс «Плоская заливка» (по умолчанию включён); дефолтное значение `inpaint-negative-prompt` со стоп-словами чернил |
| `frontend/css/styles.css` | Стиль `.inpaint-flat-fill-row` |
| `frontend/locale/{ru,en}.json` | Ключ `inpaint.assistant.flat_fill` |
| `tests/test_inpaint_ink_brush_fix.py` | **Новый файл** — 21 регрессионный тест (T-1…T-10) |
| `tests/test_prompt_gen_api.py` | Обновлено устаревшее утверждение о кэше (см. ниже) |

### 8.1. Проверка на реальном инциденте

Тот же самый ответ VLM, который вызвал дефект, пропущен через исправленный конвейер:

```
ДО:  POS: A seamless dark red and black ink-splash background filling the entire area, with dense
          crimson washes, irregular black brush strokes, dry-brush streaks, scattered droplets,
          and rough manga-style splatter matching the surrounding artwork...
      NEG: text, japanese, kanji, kana, letters, sfx, ... visible seam

ПОСЛЕ:
      POS: A seamless dark red and black background filling the entire area, and matching the
           surrounding artwork, clean smooth flat background, seamless solid fill, uniform tone
      NEG: text, japanese, kanji, kana, letters, sfx, ... visible seam, ink, brush strokes,
           dry brush, splatter, droplets, ink wash, streaks, black marks, dark spots,
           dirty shading, grunge
```

Чернильная лексика в положительном промпте: **0**. Обязательные стоп-слова в отрицательном: **все на месте**.

### 8.2. Отклонения от ТЗ (осознанные)

1. **§4.4: `no ink, no brush marks` НЕ добавлены в положительный промпт.** Причина техническая: диффузионные модели не исполняют отрицания, поэтому литерал «brush» в положительном промпте — это ровно тот триггер, который мы убираем. Подавление чернил вынесено в отрицательный промпт. Это же зафиксировано тестом (T-10 падал, пока фраза стояла в позитивном промпте).
2. **§4.3/G6: `DEFAULT_NEGATIVE_PROMPT` не изменён глобально.** Стоп-слова чернил добавлены **условно** — только для задач удаления текста/SFX. Глобальное добавление подавило бы законный лайнарт при правке контуров.
3. **Инкрементация отрицательных стоп-слов привязана к удалению текста/SFX**, а не к любому удалению: `remove the car` не должен получать `no ink` (иначе заливка будет конфликтовать с окружающим лайнартом).
4. **Порядок веток в эвристике:** `is_replace → white_bubble → is_remove → flat_color → colored_art`. Ветки `white_bubble`/`flat_color` дают корректные формулировки без чернильной лексики и остаются приоритетными; замена на «плоскую заливку» применяется к `colored_art` (именно он давал «manga brush and illustration texture»).

### 8.3. Побочная находка: устаревший тест

`tests/test_prompt_gen_api.py::test_prompt_gen_api_heuristic_fallback_and_cache` **падал до начала работ** (проверено через `git stash`). Причина — коммит `3d717cb` («disable fallback caching») изменил поведение, но тест остался с прежним утверждением `cached is True`. Тест обновлён под фактическое (и задуманное) поведение: результат fallback по ошибке VLM **не** кэшируется, а явно запрошенный эвристический режим — кэшируется (проверка обоих случаев).

### 8.4. Состояние тестов

```
python -m pytest tests/ -q   →  207 passed
```

Новый файл `tests/test_inpaint_ink_brush_fix.py` — 21 тест (T-1…T-10), покрывает все восемь найденных пробелов G1…G8.

---

## 9. Второй корневой дефект: красная маска отравляла описание VLM (обнаружен 03.11.2026, 04:49)

### 9.1. Симптом

После исправления из §8 чернильные мазки исчезли, но заливка стала **плоским однотонным красным пятном**:

```
ПРОМПТ: "A completely flat, smooth, uniform dark crimson-red background filling the entire image.
          The surface is an even solid color, clean smooth flat background, seamless solid fill, uniform tone"
```

Замеры (маска `nunchaku_mask_33272`, генерация 04:46, seed 1793663207):

| Область | mean RGB | std |
|---|---|---|
| Заливка (ядро маски) | 103.8 / 2.4 / 25.8 | **7.4** |
| Окружение | 93.1 / 21.4 / 43.1 | **35.2** |

Отношение текстуры заливки к окружению — **0.16**, цвет расходится на 19. Плоская клякса.

### 9.2. Настоящая причина: VLM описывал не фон, а красную заглушку

Реконструкция исходника (фрагменты `*_raw.png` двух генераций покрывают разные кропы, вне них источник совпадает на **100.00 %** пикселей) показала:

| Область | mean RGB | std |
|---|---|---|
| Исходник **под маской** | 166.0 / 99.0 / 100.0 | 69.9 / 111.3 / 111.2 |
| Исходник вокруг маски | 93.1 / 21.4 / 43.1 | 35.2 / 36.4 / 34.7 |

Под маской был **японский звуковой эффект — кана «ブ»** с белой обводкой и тёмно-красной заливкой.

Дальше — ключевая находка. `build_vlm_images()` (`backend/prompt_gen.py`) накладывает на маску **красную подсветку** (`_apply_red_mask_highlight`, alpha 0.45), чтобы VLM понимал, где именно заливка. Под красной подсветкой SFX превращается в **ровную красную фигуру**. VLM видел:

![VLM input](outputs/_investigation/vlm_input_crop.png)

и описывал именно её: «completely flat, smooth, uniform dark crimson-red … The surface is an even solid color». Модель не выдумывала — она буквально описывала видимую заглушку.

Проверка прямым прогоном FLUX 3 Prompt Upsampler (Nunchaku) на реальном входе пользователя:

| Вход + промпт | Ответ PUP |
|---|---|
| **A.** исходный кроп + красная подсветка, старый промпт | «A completely flat, smooth, uniform dark red background filling the entire square image…» — **воспроизводит дефект пользователя** |
| **B.** залеченный кроп + новый промпт | «A continuous abstract Japanese-style background surface… the same dense, layered field of dark crimson and burgundy foliage-like texture…» |
| **C.** залеченный кроп + новый промпт (лёгкая подсветка) | «A dense, painterly field of dark crimson and scarlet foliage fills the entire composition, continuing seamlessly through the central area…» |
| **D.** исходный кроп + новый промпт | «A seamless continuation of the dark crimson and burgundy textured background…» |

### 9.3. Исправление

1. **`build_vlm_images()` — залечить маску ДО красной подсветки.** VLM больше не видит SFX и описывает окружающий фон. Добавлена функция `_heal_masked_area()` (переиспользует `heal_masked_image` из `backend/masking.py`). Сила подсветки снижена 0.42→0.30 (обзор) и 0.45→0.22 (кроп).
2. **Переформулировать промпт: описывать окружение, а не заглушку.** «The highlighted area is an empty placeholder… describe the surface AROUND it… the same surface continues into it». Запрет «опиши плоскую однотонную заливку» снят.
3. **Сузить санитайзер.** Раньше он вырезал `texture`, `splatter`, `streaks`, `ink` — но после переформулировки VLM описывает **окружающий арт**, и слова про его текстуру нужны диффузии. Теперь вырезаются только однозначные описания удалённого элемента (`ink-splash`, `black brush strokes`, `dry-brush streaks`, `manga-style splatter`, `crimson washes`, `painted texture`).
4. **Маркер заменён**: `clean smooth flat background, seamless solid fill, uniform tone` → `seamless continuation of the surrounding background surface, matching its colour, tone and texture`.
5. **Чекбокс переименован и перевёрнут по умолчанию.** Теперь это «Плоская однотонная заливка (вместо продолжения фона)», **выключен по умолчанию**. Раньше он был включён и именно он требовал плоской заливки — то есть сам дефект был зашит в интерфейс.
6. **Негативный промпт:** убраны `dirty shading` и `grunge` — они подавляли законную живописную текстуру фона.

### 9.4. Проверка реальной генерацией

Тот же seed `1793663207`, тот же движок FLUX.2 Klein 9B (Nunchaku SVDQ), та же маска — отличается только промпт:

| | Заливка mean | Заливка std | std ratio | Δ цвета |
|---|---|---|---|---|
| Старый промпт (04:46) | 103.8 / 2.4 / 25.8 | 7.4 | **0.16** | 19.0 |
| **Новый промпт (05:09)** | 90.1 / 13.3 / 37.7 | **33.8** | **0.81** | **8.0** |

![Сравнение](outputs/_investigation/old_vs_new.png)

Слева — плоская красная клякса, справа — SFX удалён, текстура фона непрерывна и естественна.

### 9.5. Изменённые файлы (раунд 2)

| Файл | Что сделано |
|---|---|
| `backend/prompt_gen.py` | `_heal_masked_area()`; залечивание маски в `build_vlm_images()`; снижена сила красной подсветки; `flat_fill` по умолчанию `False` |
| `backend/vision_runtime.py` | Переформулированы `SYSTEM_PROMPT`, `INPAINT_PUP_BASE_RULES`, `INPAINT_PUP_REMOVAL_RULES`, `_build_inpaint_request`; сужены `_INPAINT_ARTEFACT_PATTERNS`; маркер заменён на `CONTINUATION_MARKER` |
| `backend/prompt_fallback.py` | Ветка `is_remove` — «seamless continuation of the surrounding … background surface»; из `MANDATORY_INK_NEGATIVES` убраны `dirty shading` и `grunge` |
| `frontend/index.html` | Чекбокс переименован и **сброшен** (по умолчанию продолжение фона) |
| `frontend/js/app.js` | `flatFill` по умолчанию `false` |
| `frontend/locale/{ru,en}.json` | Новый текст чекбокса |
| `tests/test_inpaint_ink_brush_fix.py` | Обновлены ожидания; добавлен **T-11** (VLM не видит удалённый элемент) |

```
python -m pytest tests/ -q   →  212 passed
```

---

## 10. Третий корневой дефект: FLUX 3 PUP — каптионер, а не ассистент по удалению (обнаружен 03.11.2026, 05:52)

### 10.1. Симптом

Пользователь нарисовал маску на японском SFX, расположенном поверх **тёмно-синего силуэта персонажа**, и получил не заливку, а шумную «торс»-текстуру с багровыми кляксами.

Промпт прогона 05:34 (`outputs/nunchaku_20261103_053433_1793666073.json`, seed `1793666073`):

```
"A seamless continuation of the surrounding dark graphic artwork fills the entire marked area:
 a deep black and burgundy character-like torso rendered in a gritty, high-contrast ink style,
 with irregular crimson brushwork spreading naturally across the torso, …"
```

Замеры (маска `nunchaku_mask_29720`):

| Область | mean RGB | std |
|---|---|---|
| Заливка (ядро маски) | 65.1 / 13.0 / 20.9 | 35.3 / 12.2 / 12.2 |
| Окружение | 70.5 / 44.0 / 64.7 | 78.6 / 74.7 / 68.0 |

std ratio — **0.27**, Δ цвета — **43.9**. Это хуже, чем «просто плоское красное» из §9: здесь появились именно чернила.

### 10.2. Ответ на вопрос пользователя

> «То есть, дело в промпте? … Upsampler не понимает, что он помощник по удалению объектов?»

**Да, дело в промпте — но не в том, который пишет человек, а в том, который машина берёт у PUP.** FLUX 3 Prompt Upsampler — это каптионер: он описывает то, что видит, и ассистентом по удалению объектов не является. Проверено измерениями (`outputs/_investigation/pup_strict_test.py`, `outputs/_investigation/pup_wording_test.py`):

1. **Ограничения длины он не выполняет.** System prompt с явным правилом «Answer with AT MOST 12 words» дал **41 слово**.
2. **Он каптионирует весь кроп, а не область вокруг маски.** На кропе пользователя он описал «dark navy-black curved forms along the lower sides» (силуэт персонажа) и «a narrow red vertical form near the top centre» (столб брызг) — то есть всю панель, а не дырку.
3. **Запрет формулировки он соблюдает частично.** Правило «NEVER describe characters, faces, hair, or clothing» не помешало ему написать «character-like torso».

Механика дефекта прямая: подпись PUP → `parse_vlm_response_text()` → промпт диффузии → диффузия исполняет промпт **буквально**. Слово `brushwork` в промпте = мазок кисти в картинке; слова `character-like torso` = нарисованный торс поверх удаляемого SFX.

Практический вывод: **ограничить PUP словами в промпте нельзя — он их игнорирует.** Защита должна быть в коде, который решает, что попадёт в промпт диффузии.

### 10.3. Что изменено: промпт больше не доверяет подписи

Новый модуль `backend/surface_profile.py`:

| Функция | Что делает |
|---|---|
| `measure_surface()` | Измеряет **узкую кайму (≈8 px) сразу за границей маски**: цвет, насыщенность, робастную текстуру. Кайма, а не широкое кольцо: на кейсе пользователя широкое кольцо давало «красный» цвет, потому что вокруг торса лежали красные брызги и белая бумага |
| `rgb_to_color_name()` | Тёмный насыщенный цвет больше не «black»: `#010320` → **dark navy blue** (иначе заливка получила бы неверный цвет) |
| `caption_is_removal_safe()` | Подпись **не используется**, если в ней есть чернила/кисть, люди и части тела, «плоская/однотонная» формулировка или жёсткие стилевые ярлыки (`gritty`, `high-contrast`) |
| `caption_colour_conflicts()` | Подпись, называющая **чужое цветовое семейство** («deep burgundy-red … near the top» для синей панели), отвергается: иначе диффузия покрасит дырку чужим цветом |
| `extract_surface_vocabulary()` | Белый список слов о материале поверхности (`dense painterly texture`, `mottled shading`, `fine grain`, …) — единственный способ, которым подпись вообще может влиять на промпт |
| `build_removal_prompt()` | Сборка: безопасная и согласованная подпись идёт **дословно** (в ней фактура текстуры, нужная диффузии), иначе — измеренная фраза + белый список. Маркер продолжения добавляется всегда |

Сопутствующие правки раунда 3:

* `max_new_tokens` **90 → 160** (`PUP_MAX_NEW_TOKENS`) — подпись обрывалась на полуслове («The central region continues as»), и диффузия получала незаконченный промпт;
* стоп-слова чернил в negative добавляются для **любого** удаления, а не только при явном «text/SFX» (раньше «удали это» теряло защиту);
* из подписи вырезаются клаузы про удаляемый элемент («the area formerly covered by the Japanese SFX placeholder») и отрицания («with no figures, Japanese sound-effect graphics, …») — и то и другое провоцировало диффузию наоборот.

### 10.4. Проверка реальной генерацией

Тот же движок FLUX.2 Klein 9B (Nunchaku SVDQ), те же маски, те же seed'ы — отличается только сборка промпта.

**Кейс B — SFX на синем силуэте (seed `1793666073`)**

| | Заливка mean | Заливка std | std ratio | Δ цвета | тёмных пикселей |
|---|---|---|---|---|---|
| Промпт пользователя (05:34) | 65.1 / 13.0 / 20.9 | 35.3 / 12.2 / 12.2 | 0.27 | 43.9 | — |
| **Новый промпт (06:28)** | 22.7 / 28.2 / 51.4 | 1.5 / 1.2 / 1.1 | 0.018 | 47.4 | **0 %** |

Промпт, который ушёл в диффузию: `smooth dark navy blue surface, mottled shading, seamless continuation of the surrounding surface, matching its colour, tone and texture`. Подпись PUP на этом кропе была отвергнута проверкой согласованности цвета — и это правильное поведение: в ней было описание посторонних участков панели.

Δ цвета здесь велик **по построению метрики**: «окружение» — это смешанная панель (красные брызги + белая бумага + синий силуэт), а заливка корректно продолжает именно синюю поверхность, к которой прилегала маска.

**Кейс A — SFX на красном фоне (seed `1793663207`)**

| | Заливка mean | Заливка std | std ratio | Δ цвета |
|---|---|---|---|---|
| Плоская красная клякса пользователя (04:46) | 103.8 / 2.4 / 25.8 | 7.4 | 0.16 | 19.0 |
| Лучший результат раунда 2 (05:09) | 90.1 / 13.3 / 37.7 | 33.8 | **0.81** | **8.0** |
| **Новый промпт (06:27)** | 113.0 / 21.9 / 37.2 | 29.5 | 0.33 | 15.5 |

![Кейс A](outputs/_investigation/round3_caseA.png)

Порядок слева направо: исходник → плоская красная клякса (04:46) → лучший результат раунда 2 (05:09) → 06:20 → 06:27 (новый промпт).

**Честный итог по кейсу A:** дефект устранён (SFX удалён, фактура панели продолжается, никаких чернил и никакой кляксы), но текстура заливки воспроизводится **грубее**, чем в удачном прогоне раунда 2 (std ratio 0.33 против 0.81), и заливка чуть светлее окружения. Причина — сама подпись: PUP в этом прогоне дал развёрнутое перечисление («mottled crimson and burgundy texture with fine granular speckling…»), тогда как удачный промпт раунда 2 был коротким и категоричным («a dense, painterly field of dark crimson and scarlet … fills the entire composition»). Подпись PUP не детерминирована, поэтому совпадение с идеалом раунда 2 не гарантировано — зато теперь **гарантировано отсутствие галлюцинаций** в промпте.

![Кейс B](outputs/_investigation/round3_caseB.png)

Порядок слева направо: исходник → результат пользователя 05:34 (шумный «торс» с багровыми кляксами) → 06:09 → 06:28. SFX удалён, синяя поверхность продолжается, артефактов нет.

### 10.5. Побочная находка: «реконструкция исходника» из `*_raw.png` была неверной

Файл `outputs/masks/*_raw.png` — это **холст после генерации** (под маской лежит результат модели, а не исходные пиксели), а не исходник. Проверено замером: под маской прогона 05:34 лежит результат 05:34, а не SFX. Настоящий исходник — загрузка пользователя `outputs/uploads/ref_34dd578b.png` (2800×4021); под маской в ней действительно белый глиф с тёмно-красной заливкой (case A) и белый глиф на синем силуэте (case B).

### 10.6. Побочно найденный баг: эвристика не считала «delete» удалением

Живая проверка `POST /api/inpaint/prompt` (эвристический режим) на кейсе пользователя выдала:

```
"seamless dark blue and red background, matching surrounding dark blue,
 manga brush and illustration texture, clean edges, no text"
```

Причина: `_RE_REMOVE` в `backend/prompt_fallback.py` знал `remov*`, `eras*`, `clear*`, `clean*`, но **не знал `delet*`** — английское «delete this Japanese SFX» не считалось удалением, инструкция уходила в общую ветку `colored_art`, и та вставляла «manga brush and illustration texture» (тот самый триггер чернил) плюс отрицание «no text». VLM-ветка при этом работала верно: там `_is_removal_instruction` содержит `delete`. Регексы приведены к общему виду, добавлен тест **T-17**.

### 10.7. Изменённые файлы (раунд 3)

| Файл | Что сделано |
|---|---|
| `backend/surface_profile.py` | **Новый модуль**: измерение поверхности, именование цвета, безопасность и согласованность подписи, сборка removal-промпта |
| `backend/vision_runtime.py` | `PUP_MAX_NEW_TOKENS = 160`; расширены `_INPAINT_META_PATTERNS` (удаляемый элемент + отрицания); стоп-слова чернил для любого удаления; сборка removal-промпта через `build_removal_prompt`; system prompt запрещает называть субъектов |
| `backend/prompt_gen.py` | Измерение поверхности (`measure_surface`) и передача профиля в `generate_vlm_prompt`; `PROMPT_TEMPLATE_VERSION` → `2026-11-03-ink-brush-fix-v2` |
| `backend/prompt_fallback.py` | Эвристика использует тот же измеренный профиль; `_RE_REMOVE` дополнен `delet*`/`wip*`/`scrub*`; `extract_dominant_colors`/`rgb_to_color_name` перенесены в `surface_profile` (обратный экспорт сохранён) |
| `tests/test_inpaint_ink_brush_fix.py` | Добавлены **T-12…T-17**: измерение по кайме, тёмный насыщенный цвет ≠ black, устойчивость к выбросам, безопасная подпись идёт дословно, конфликт цвета отвергается, вырезание упоминаний удаляемого элемента, английское «delete» распознаётся как удаление |
| `tests/test_prompt_fallback.py` | Устаревшее ожидание «ink splash в позитивном промпте» заменено на запрет |

Живая проверка API на кейсе пользователя (сервер с новым кодом):

```
эвристика : "smooth dark navy blue surface, seamless continuation of the surrounding surface, …"
живой PUP : "smooth dark navy blue surface, mottled shading, seamless continuation of the surrounding surface, …"
измерение : {"primary": "dark navy blue", "primary_hex": "#010320", "texture": "smooth"}
```

```
python -m pytest tests/ -q   →  227 passed
```

