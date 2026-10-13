# LazySD-UI

[![Release: Pre-Alpha](https://img.shields.io/badge/Release-Pre--Alpha-orange.svg)](https://github.com/nnmgrp/LazySD-UI)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%2013.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**LazySD-UI** — легковесный автономный WebUI для быстрого запуска квантованных диффузионных моделей генерации изображений. 

Построен на **FastAPI + uvicorn** с кастомным интерфейсом на чистом HTML/CSS/JS (тёмная тема, IDE-grid layout). **Никаких узлов и спагетти ComfyUI, никакого громоздкого Gradio** — только минималистичный интерфейс: выбор модели, ввод промпта, регуляторы и мгновенная генерация.

> **Статус проекта: Pre-Alpha.**  
> Проект создан чисто в рамках изучения вайб-кодинга (vibe-coding) с помощью ИИ-агентов и для решения собственных прикладных задач.

---

## ⚠️ Важный правовой дисклеймер о QuantFunc

1. **Репозиторий НЕ предоставляет и НЕ распространяет программы или бинарники QuantFunc.** Мы не обладаем лицензиями на их закрытое ПО и не несём за него ответственности.
2. Поддержка QuantFunc в проекте — это исключительно **вайб-код обёртка (subprocess wrapper)** для тех, кто не хочет возиться с ComfyUI.
3. **Планы по отказу от QuantFunc:** Скорее всего, в будущих обновлениях поддержка QuantFunc будет **полностью удалена** из-за:
   - Невозможности писать промпты на других языках, кроме английского (закрытый C++ пайплайн QuantFunc отбрасывает или ломает мультиязычные токены);
   - Сырых и нестабильных весов с частыми внутренними ошибками шейпинга (`TensorShape index out of range`);
   - Закрытой лицензионной политики и непрозрачности C++ бинарников.
4. Основной вектор развития проекта направлен на прозрачные и открытые решения: **stable-diffusion.cpp (GGUF)** и **SDNQ / Diffusers**.

---

## Поддерживаемые бэкэнды и модели

| Движок / Рантайм | Модели | Формат | Особенности |
|---|---|---|---|
| **SD.CPP (stable-diffusion.cpp)** | **Qwen-Image-2.1 GGUF**, **Flux.1 Schnell**, **SDXL Turbo** | GGUF (Q4_K_M, Q4_0, Q4_K_S) | Чистый C++/CUDA, мультиязычные открытые энкодеры (Qwen3-VL-8B-Instruct), высокая стабильность. |
| **SDNQ (PyTorch + Diffusers)** | **Qwen-Image-2.1 Turbo (SDNQ INT8)** | INT8 / SVDQuant | Нативный 3-этапный запуск с полной изоляцией VRAM (Text Encoder → del → Transformer → del → VAE), помещается в 16 ГБ VRAM без сбоев драйвера и без полос VAE. Поддерживает и обычные веса Diffusers. |
| **QuantFunc (Legacy Wrapper)** | **Flux.2 Klein 9B / 4B**, **Z-Image Turbo** | NVFP4 / INT4 Lighting | Закрытый C++ CLI (сторонний). Экспериментальный режим, планируется к выводу из проекта. |

---

## Основные возможности

- **Мультибэкэндность**: бесшовное переключение между GGUF (`sd.cpp`), PyTorch/Diffusers (`sdnq`) и legacy CLI (`quantfunc`).
- **Тонкие настройки генерации**: регуляторы CFG (включая True CFG Scale для Qwen), Noise / $\eta$, Steps, выбор сэмплеров (Euler, Euler Ancestral, LCM, DPM++ 2M, UniPC, Heun).
- **Img2Img и референсы**: встроенный drag-and-drop с регулировкой Denoising Strength.
- **Выбор VAE**: поддержка оригинального VAE и Texture-Fix VAE (madebyollin) с автоматической конвертацией стейт-диктов.
- **Галерея и история**: сохранение параметров в метаданных PNG и JSON-sidecar с возможностью повторного применения промпта в один клик.
- **Двуязычный интерфейс**: поддержка русского и английского языков (автоопределение по браузеру).

---

## Быстрый старт

### Требования
- **ОС**: Windows 10/11 x64 (основная платформа) или Linux x64
- **Видеокарта**: NVIDIA RTX 20xx / 30xx / 40xx / 50xx (от 8–12 ГБ VRAM)
- **Драйвер NVIDIA**: $\ge$ 560
- **Python**: 3.11 или 3.12

### Запуск на Windows

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/nnmgrp/LazySD-UI.git
   cd LazySD-UI
   ```
2. Запустите лаунчер:
   ```cmd
   start.bat
   ```
   Скрипт автоматически создаст виртуальное окружение, установит зависимости и откроет WebUI в браузере по адресу: **http://127.0.0.1:7860**.

### Ручной запуск

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

---

## Структура репозитория

```text
LazySD-UI/
├── app.py                      # FastAPI сервер и REST API
├── requirements.txt            # Зависимости Python
├── start.bat                   # Лаунчер для Windows
├── backend/
│   ├── worker.py               # Оркестратор задач генерации
│   ├── sd_cpp.py               # Интеграция с stable-diffusion.cpp (GGUF)
│   ├── sdnq_engine.py          # 3-этапный изолированный движок SDNQ / Diffusers
│   ├── flowmatch_samplers.py   # FlowMatch сэмплеры и интерполяция sample_sigmas
│   ├── gpu.py                  # Автоопределение GPU и SM-архитектуры
│   ├── security.py             # Валидация путей и защита от инъекций
│   └── registry/               # JSON-спецификации моделей и пресетов
├── frontend/                   # Кастомный UI (HTML, CSS, JS, i18n)
└── tests/
    └── smoke_test.py           # Автотесты целостности (CPU-only)
```

---

## Лицензия

Исходный код проекта распространяется под лицензией **GNU General Public License v3.0 (GPLv3)**. Подробнее см. в файле [LICENSE](LICENSE).

Веса моделей и внешние бинарники принадлежат их соответствующим авторам и регулируются их собственными лицензиями.
