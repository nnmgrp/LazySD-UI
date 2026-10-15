/* ============================================================
   Dropzone — drag & drop + paste handler for image inputs
   Reuses pattern from HF Space prithivMLmods/Flux.2-Klein-KV-Edit
   ============================================================ */

class Dropzone {
  constructor(element, options = {}) {
    this.element = element;
    this.onFile = options.onFile || (() => {});
    this.acceptedTypes = options.acceptedTypes || ["image/png", "image/jpeg", "image/webp"];
    this.maxSizeMb = options.maxSizeMb || 25;

    this.dropOverlay = null;
    this._setup();
  }

  _setup() {
    // Global drop overlay (covers entire preview area)
    this.dropOverlay = document.createElement("div");
    this.dropOverlay.className = "drop-overlay";
    this.dropOverlay.textContent = "Drop image here";
    this.element.appendChild(this.dropOverlay);

    // Drag&drop listeners
    let dragCounter = 0;

    this.element.addEventListener("dragenter", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter++;
      this.dropOverlay.classList.add("active");
    });

    this.element.addEventListener("dragleave", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        this.dropOverlay.classList.remove("active");
      }
    });

    this.element.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.stopPropagation();
      e.dataTransfer.dropEffect = "copy";
    });

    this.element.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter = 0;
      this.dropOverlay.classList.remove("active");

      const files = Array.from(e.dataTransfer.files);
      const imageFile = files.find(f => this.acceptedTypes.includes(f.type));
      if (imageFile) {
        this._handleFile(imageFile);
      }
    });

    // Paste handler (Ctrl/Cmd+V) — only when this element is active
    document.addEventListener("paste", (e) => {
      const items = Array.from(e.clipboardData?.items || []);
      const imageItem = items.find(item => item.type.startsWith("image/"));
      if (imageItem) {
        const file = imageItem.getAsFile();
        if (file) {
          this._handleFile(file);
        }
      }
    });

    // Click to browse
    if (this.element.classList.contains("clickable")) {
      this.element.addEventListener("click", () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = this.acceptedTypes.join(",");
        input.onchange = () => {
          if (input.files[0]) {
            this._handleFile(input.files[0]);
          }
        };
        input.click();
      });
    }
  }

  _handleFile(file) {
    // Validate type (MIME type or extension fallback)
    const ext = "." + (file.name ? file.name.split(".").pop().toLowerCase() : "");
    const validExts = [".png", ".jpg", ".jpeg", ".webp"];
    const isMimeValid = this.acceptedTypes.includes(file.type);
    const isExtValid = validExts.includes(ext);

    if (!isMimeValid && !isExtValid) {
      console.warn(`[dropzone] Rejected file type: ${file.type || ext}`);
      if (window.App && typeof window.App.toast === "function") {
        window.App.toast("error", `Unsupported file type: ${file.type || ext}. Allowed: PNG, JPEG, WEBP`);
      }
      return;
    }
    // Validate size
    const sizeMb = file.size / 1024 / 1024;
    if (sizeMb > this.maxSizeMb) {
      console.warn(`[dropzone] File too large: ${sizeMb.toFixed(1)} MB > ${this.maxSizeMb} MB`);
      if (window.App && typeof window.App.toast === "function") {
        window.App.toast("error", `File too large: ${sizeMb.toFixed(1)} MB (Max: ${this.maxSizeMb} MB)`);
      }
      return;
    }

    // Read as data URL for preview
    const reader = new FileReader();
    reader.onload = (ev) => {
      this.onFile({
        file: file,
        dataUrl: ev.target.result,
        name: file.name,
        size: file.size,
        type: file.type,
      });
    };
    reader.readAsDataURL(file);
  }
}

// Export
window.Dropzone = Dropzone;
