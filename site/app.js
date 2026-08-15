const copyButton = document.querySelector("[data-copy-target]");
const copyStatus = document.querySelector("#copy-status");

if (copyButton && copyStatus) {
  copyButton.addEventListener("click", async () => {
    const targetId = copyButton.dataset.copyTarget;
    const target = targetId ? document.getElementById(targetId) : null;

    if (!target) {
      copyStatus.textContent = "Command unavailable.";
      return;
    }

    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      copyButton.classList.add("is-copied");
      copyButton.querySelector(".copy-label").textContent = "Copied";
      copyStatus.textContent = "Run command copied to clipboard.";

      window.setTimeout(() => {
        copyButton.classList.remove("is-copied");
        copyButton.querySelector(".copy-label").textContent = "Copy";
      }, 1800);
    } catch {
      copyStatus.textContent = "Clipboard access was blocked. Select and copy the command manually.";
      target.focus?.();
    }
  });
}
