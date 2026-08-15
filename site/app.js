const copyButton = document.querySelector("[data-copy-target]");
const copyStatus = document.querySelector("#copy-status");

if (copyButton && copyStatus) {
  copyButton.addEventListener("click", async () => {
    const target = document.getElementById(copyButton.dataset.copyTarget ?? "");
    if (!target) return;

    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      copyButton.classList.add("is-copied");
      copyButton.querySelector(".copy-label").textContent = "Copied";
      copyStatus.textContent = "Run command copied to clipboard.";
      window.setTimeout(() => {
        copyButton.classList.remove("is-copied");
        copyButton.querySelector(".copy-label").textContent = "Copy command";
      }, 1800);
    } catch {
      copyStatus.textContent = "Select the command and copy it manually.";
    }
  });
}

const revealItems = document.querySelectorAll("[data-reveal]");

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.14 },
  );

  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}
