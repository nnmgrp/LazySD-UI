# ТЕХНИЧЕСКОЕ ЗАДАНИЕ (ТЗ) ДЛЯ GOOGLE AI STUDIO / GEMINI BUILDER
## Прототип: «LazySD Manga Studio & Chat Workbench» на базе nnmgrp/GenTonik

---

> **Инструкция для Builder в Google Studio:**  
> Ты — Senior Full-Stack & Graphics Web Architect. Твоя задача — создать интерактивный веб-прототип **LazySD Manga Studio**, объединяющий Photoshop-подобный послойный редактор манги/вебтунов и чат-режим генерации (Midjourney/ChatGPT-style feed).  
> В качестве графического фундамента используй архитектуру, математику трансформаций и систему слоёв из **`nnmgrp/GenTonik`** (React 19 + TypeScript + Vite + Tailwind CSS v4 + WebGL2/Canvas2D).  
> Все AI-операции (диффузия, инпейнт, замер поверхности каймы, растрирование) подключаются через типизированные точки входа (API-адаптеры) к локальному серверу **Lazy SD UI**.

---

## 1. РЕФЕРЕНСНАЯ АРХИТЕКТУРА ИЗ GENTONIK (`nnmgrp/GenTonik`)

В репозитории `nnmgrp/GenTonik` уже реализована стабильная база послойного графического редактора. Этот код и математические контракты берутся за основу:

### 1.1. Стек технологий
- **Frontend Core:** React 19, TypeScript 5.9, Vite 7, Tailwind CSS v4 (`@tailwindcss/vite`).
- **Рендеринг:** Двойной пайплайн:
  1. *WebGL2 FBO Compositor* (`src/webgl/`): Аппаратный рендеринг скринтонов, шейдеров и наложения слоёв на GPU.
  2. *Canvas2D Fallback* (`src/composite.ts`): Интеллектуальный обход для сверхтяжелых документов (сканы $12000 \times 17000$ px), превышающих лимит видеокарты `MAX_TEXTURE_SIZE`.
- **Трансформация:** `src/canvas-controls/` и `src/homography.ts` — полноценный Gizmo-манипулятор с поддержкой сдвига, масштабирования, вращения, 2D-скоса (Krita-style Skew) и 4-точечной перспективной гомографии (Perspective Transform).
- **Экспорт/Импорт:** `src/ora-format.ts` — поддержка формата OpenRaster (`.ora`) с упаковкой слоёв и масок через JSZip.

### 1.2. Базовые типы данных (`src/types.ts`)
```typescript
export type LayerType = 
  | 'image'        // Растровый скан / исходник манги / фон
  | 'screentone'    // Процедурный скринтон (точки, линии, штриховка, LPI)
  | 'solid'        // Сплошная заливка цветом
  | 'transparent'  // Пустой прозрачный контейнер под маски/кисть
  | 'text'         // Текстовый слой (диалоги манги, тайпсет)
  | 'vector'       // Векторные панели, рамки, скоростные линии
  | 'inpaint';     // [НОВОЕ] Сгенерированный фрагмент удаления с мягкой маской

export type BlendMode = 
  | 'normal' 
  | 'multiply'     // Критично для скринтонов и контуров манги
  | 'screen' 
  | 'overlay' 
  | 'darken' 
  | 'lighten';

export interface Vec2 {
  x: number;
  y: number;
}

export interface LayerTransform {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number;     // Градусы
  skewX: number;        // Сдвиг по X
  skewY: number;        // Сдвиг по Y
  corners?: [Vec2, Vec2, Vec2, Vec2] | null; // 4 угла гомографии (Perspective)
}

export interface ScreentoneParams {
  patternType: 'dots' | 'lines' | 'crosshatch' | 'noise' | 'hexgrid' | 'stipple';
  dotShape: 'circle' | 'square' | 'diamond' | 'hexagon';
  dotSize: number;
  spacingX: number;
  spacingY: number;
  density: number;
  angle: number;
  frequency: number;    // LPI (Lines Per Inch)
  unit: 'px' | 'mm' | 'in' | 'lpi';
  colorPattern: string; // Обычно #000000
  colorBg: string;      // Обычно #ffffff или transparent
  seamless: boolean;
  gradType: 'none' | 'linear' | 'radial';
  gradAngle: number;
}
```

---

## 2. ЧТО НЕОБХОДИМО ДОБАВИТЬ В ПРОТОТИП

Прототип объединяет две взаимосвязанные рабочие среды:
1. **Режим 1: Chat/Feed Mode (Генерация)** — быстрый диалоговый интерфейс.
2. **Режим 2: Unified Manga Studio (Photoshop-like)** — детальная послойная студия (GenToniK + Inpaint + Halftone + Typeset).

```
┌────────────────────────────────────────────────────────────────────────┐
│                              TOP NAVBAR                                │
│ [ 💬 Чат генерации ]   [ 🎨 Студия (GenToniK Studio) ]   [ ⚙ Настройки ]│
├───────────────────────────────────┬────────────────────────────────────┤
│         РЕЖИМ 1: CHAT FEED        │       РЕЖИМ 2: MANGA STUDIO        │
│                                   │                                    │
│ • Линейная лента итераций         │ • Табы документов (Photoshop style)│
│ • Карточки генераций:             │ • Палитра инструментов (Brush/Wand)│
│   - Превью и статус               │ • WebGL2 холст с зумом и линейками │
│   - Метаданные (seed, cfg, steps) │ • Менеджер слоёв:                  │
│   - Чипсы быстрых действий:       │   - [Text] Тайпсет диалогов        │
│     [ 🎨 Открыть в Студии ]        │   - [Ink] Лайнарт и чистка         │
│     [ 🪄 Удалить SFX / Inpaint ]   │   - [Tone] Скринтоны (Halftone GT) │
│     [ 🔄 Вариация / Реролл ]       │   - [Inpaint] Результаты удаления  │
│ • Строка ввода промпта внизу      │   - [Base] Исходный скан (Lock)    │
│   с быстрыми переключателями      │ • Панель свойств активного слоя    │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

### 2.1. Дополнения для Студии (Photoshop-like Canvas)

#### А. Слой Inpaint (`InpaintLayer`)
- Хранит сгенерированный растровый фрагмент (`image`) и соответствующую альфа-маску (`maskImage`).
- Поддерживает прозрачность (Opacity) и включение/выключение видимости (глазок).
- При рендеринге фрагмент отсекается по маске (`globalCompositeOperation = 'destination-in'`) и накладывается поверх оригинального скана без деструктивного затирания исходника.

#### Б. Инструменты выделения и маскирования (Selection & Masking)
- **Brush (Кисть маски):** Настраиваемый диаметр ($1..256$ px) и мягкость (Softness $0..100\%$, гауссово сглаживание краев).
- **Eraser (Ластик маски):** Стирание нарисованной маски.
- **Magic Wand (Волшебная палочка):** Выделение связной области по порогу яркости/цвета (Flood Fill по альфе/яркости) — незаменимо для мгновенного захвата контуров баллонов и звуков SFX.
- **Lasso & Rect Marquee:** Ручное геометрическое выделение.

#### В. Тайпсеттинг и Текстовый слой (`typeset`)
- Ввод горизонтального и вертикального (`writing-mode: vertical-rl`) текста для манги.
- Выбор шрифтов, выравнивание, размер кегля, трекинг.
- Автоматическая обводка (Stroke / Outline) текста белым цветом с настраиваемой толщиной (стандарт оформления реплик в манге).
- Опция *«Fit to Bubble»*: адаптация размера и переносов строк под границы маски баллона.

#### Г. Интеграция Halftone Studio GT в слои
- Возможность превратить любой слой со скринтоном в динамический объект с параметрами из `ScreentoneParams`:
  - Выбор частоты (LPI: 30, 45, 60, 85);
  - Форма точки (круг, ромб, эллипс, линия);
  - Угол наклона ($45^\circ$ для классической манги, $0^\circ$, $90^\circ$);
  - Быстрое наложение скринтона кистью по маске.

---

### 2.2. Дополнения для Чат-Режима (Chat/Feed Generation Mode)

Вместо громоздких дашбордов с десятками ползунков на одном экране:
1. **Лента сообщений и генераций:**
   - Каждое сообщение пользователя отображает отправленный позитивный и негативный промпт.
   - Ответ сервера — интерактивная карточка с миниатюрой (WebP 384px) и полноразмерным изображением.
   - Под карточкой находятся кнопки:
     - 🎨 **«Открыть в Студии»** $\rightarrow$ переключает таб в Студию, загружает изображение как `Base Layer` нового документа.
     - 🪄 **«Удалить SFX / Текст»** $\rightarrow$ открывает диалог быстрого выделения и инпейнта.
     - 🔄 **«Реролл»** $\rightarrow$ генерация с тем же промптом и новым сидом.
     - 📋 **«Копировать метаданные»** $\rightarrow$ JSON со всеми параметрами (seed, cfg, steps, lora).
2. **Нижний плавающий Input Bar:**
   - Многострочное поле ввода промпта.
   - Быстрые селекторы в виде тегов/чипсов: Модель (`Flux.2 Klein 9B`, `FLUX.3`), Разрешение (`1024x1024`, `A4 Manga`), Бэкэнд (`Nunchaku SVDQ`, `SD.CPP`).

---

## 3. ТОЧКИ ВХОДА (API HOOKS) ДЛЯ ИНТЕГРАЦИИ С LAZY SD UI

Клиентская часть прототипа должна обращаться к бэкенду Lazy SD UI через единый типизированный интерфейс `LazySDClient`.

### 3.1. Типизированный клиент API (`src/services/lazy-sd-api.ts`)

```typescript
export interface GenerateRequest {
  prompt: string;
  negative_prompt?: string;
  model_id: string;
  variant_id?: string;
  backend?: 'nunchaku' | 'sd_cpp' | 'orbitquant' | 'quantfunc';
  width: number;
  height: number;
  steps: number;
  cfg: number;
  seed?: number;
  // Параметры для Inpaint (если с маской):
  init_image?: string;         // Base64 или путь к загруженному файлу
  mask_image?: string;         // Base64 маски (белый = изменить, черный = оставить)
  mask_strength?: number;      // 1.0 = полная перерисовка
  crop_box?: [number, number, number, number]; // [x0, y0, x1, y1] Bounding box маски
}

export interface InpaintPromptRequest {
  image_path: string;          // Путь к файлу на сервере
  mask_path?: string;
  crop_box?: [number, number, number, number];
  instruction: string;         // Например: "delete this Japanese SFX"
  vision_model?: string;       // "flux3-pup-nunchaku" | "qwen2.5vl:3b"
  vision_provider?: string;    // "nunchaku" | "ollama" | "heuristic"
  use_vlm?: boolean;
  flat_fill?: boolean;         // true = плоская заливка без сложной фактуры
}

export interface InpaintPromptResponse {
  prompt: string;
  negative_prompt: string;
  measured_profile?: {
    primary_color: string;     // Например: "dark navy blue"
    primary_hex: string;       // Например: "#010320"
    texture: 'smooth' | 'mottled' | 'coarse';
  };
  source: 'vlm' | 'heuristic';
}

export interface JobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'complete' | 'error' | 'cancelled';
  progress: number;            // 0.0 .. 1.0
  image_path?: string;         // Относительный путь к итоговому файлу
  raw_image_path?: string;     // Сырой сгенерированный фрагмент маски (_raw.png)
  error?: string;
}
```

### 3.2. Точки входа (Endpoints)

| Метод и Эндпоинт | Назначение | Точка входа в UI |
|---|---|---|
| `POST /api/upload` | Загрузка изображения (PNG/JPEG/WebP) на сервер | Drag & Drop на холст или открытие файла |
| `POST /api/generate` | Запуск задачи генерации / инпейнта | Кнопка «Отправить» в чате или «Сгенерировать в маске» |
| `GET /api/jobs/{job_id}` | Опрос статуса генерации (polling 200..400 мс) | Прогресс-бар на карточке генерации |
| `POST /api/inpaint/prompt` | Вызов VLM Помощника промпта с алгоритмическим профилированием каймы (`surface_profile.py`) | Кнопка «🪄 Сгенерировать промпт» в Inpaint |
| `GET /api/history` | Получение истории генераций | Лента чата и галерея |
| `GET /api/history/thumb/{filename}` | Быстрые миниатюры WebP 384px | Карточки в ленте чата (мгновенная загрузка) |
| `POST /api/halftone/process` | Серверный высокоточный процессинг скринтонов для гигантских сканов (600 DPI) | Меню «Запечь скринтон (HQ Server)» |
| `GET /api/models` | Получение доступных моделей, квантований и поддерживаемых фич (`supports_mask`) | Селектор моделей в строке ввода чата |

---

## 4. СТРУКТУРА КОМПОНЕНТОВ ПРОТОТИПА

```
src/
├── components/
│   ├── chat-feed/
│   │   ├── ChatFeed.tsx             // Контейнер ленты сообщений
│   │   ├── ChatMessageCard.tsx      // Карточка генерации с превью и чипсами
│   │   ├── ChatPromptInput.tsx      // Плавающая строка ввода промпта с параметрами
│   │   └── QuickInpaintModal.tsx    // Всплывающее мини-окно быстрого инпейнта
│   │
│   ├── studio/
│   │   ├── StudioWorkspace.tsx      // Главный контейнер Photoshop-подобного редактора
│   │   ├── DocumentTabBar.tsx       // Вкладки открытых документов
│   │   ├── StudioToolbar.tsx        // Вертикальная панель инструментов (Brush, Wand, Marquee, Text)
│   │   ├── StudioCanvasArea.tsx     // Холст (WebGL2 / Canvas2D) с зумом и панорамированием
│   │   ├── LayersPanel.tsx          // Панель слоёв (видимость, порядок, режимы наложения, непрозрачность)
│   │   ├── InpaintControls.tsx      // Панель инпейнта (вызов VLM промпта, выбор модели, запуск)
│   │   ├── HalftoneControls.tsx     // Панель скринтонов (LPI, угол, форма точки из GenToniK)
│   │   └── TypesetControls.tsx      // Панель типографики (шрифты, вертикальный текст, обводка)
│   │
│   └── common/
│       ├── Rulers.tsx               // Линейки документа из GenToniK
│       └── CustomScrollbars.tsx     // Скроллбары холста из GenToniK
│
├── services/
│   ├── lazy-sd-api.ts               // HTTP-клиент к Lazy SD UI бэкенду
│   ├── websocket-bridge.ts          // Опциональный сокет для мгновенного прогресса
│   └── storage.ts                   // Локальный кэш пресетов и черновиков
│
├── store/
│   ├── useStudioStore.ts            // Состояние документов, активного слоя, масок
│   └── useChatStore.ts              // Состояние ленты генераций, промптов и очереди задач
│
└── webgl/                           // Ядро рендеринга из GenToniK
    ├── composite-gl.ts
    ├── gl-shaders.ts
    └── gl-context.ts
```

---

## 5. ПОШАГОВЫЙ ПЛАН РЕАЛИЗАЦИИ ДЛЯ BUILDER

1. **Шаг 1: Инициализация и Каркас UI**
   - Развернуть макет с верхней навигацией для переключения между `[ 💬 Чат Генерации ]` и `[ 🎨 Студия ]`.
   - Подключить Tailwind v4 и тему оформления (Dark Studio Theme: фон `#121214`, акцент фиолетовый `#7c3aed`).
2. **Шаг 2: Перенос Ядра GenToniK**
   - Интегрировать `src/types.ts`, `src/composite.ts`, `src/webgl/` и манипулятор `transform-overlay-canvas.tsx`.
   - Запустить отрисовку базового холста со слоями, зумом и панорамированием.
3. **Шаг 3: Реализация Слоя Inpaint и Инструментов Маскирования**
   - Добавить тип слоя `'inpaint'`.
   - Реализовать инструменты рисования маски (Brush, Soft Eraser, Flood-fill Magic Wand).
   - Создать панель `InpaintControls` с кнопками «Сгенерировать промпт (VLM)» и «Сгенерировать в маске».
4. **Шаг 4: Реализация Chat Feed Режима**
   - Создать ленту карточек сообщений с отображением параметров и статуса генерации.
   - Реализовать кнопку *«Открыть в Студии»*, которая переключает таб и передаёт сгенерированное изображение как `Base Layer`.
5. **Шаг 5: Подключение Модуля API к Lazy SD UI**
   - Реализовать класс `LazySDClient` с методами `uploadImage`, `startGeneration`, `pollJob`, `getInpaintPrompt`.
   - Добавить мок-режим (Mock Mode) для тестирования в браузере без запущенного локального бэкенда.

---

> **Результат работы:**  
> Builder должен сгенерировать полностью рабочий, модульный код прототипа на React + TypeScript, готовый к сборке через `npm run dev` / `npm run build`.
