import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays output caption text
app.registerExtension({
  name: "Img2TxtNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "img2txt BLIP/Llava Multimodel Tagger") {
      function populate(message) {
        console.log("message", message);
        console.log("message.text", message.text);

        const insertIndex = this.widgets.findIndex((w) => w.name === "output_text");
        if (insertIndex !== -1) {
          for (let i = insertIndex; i < this.widgets.length; i++) {
            this.widgets[i].onRemove?.();
          }
          this.widgets.length = insertIndex;
        }

        const outputWidget = ComfyWidgets["STRING"](
          this,
          "output_text",
          ["STRING", { multiline: true }],
          app
        ).widget;
        outputWidget.inputEl.readOnly = true;
        outputWidget.inputEl.style.opacity = 0.6;
        outputWidget.value = message.text.join("");

        requestAnimationFrame(() => {
          const size_ = this.computeSize();
          if (size_[0] < this.size[0]) {
            size_[0] = this.size[0];
          }
          if (size_[1] < this.size[1]) {
            size_[1] = this.size[1];
          }
          this.onResize?.(size_);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        populate.call(this, message);
      };
    }
  },
});
