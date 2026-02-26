import os
import re
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from typing import Optional

# =========================
# CONFIG: AJUSTE AQUI
# =========================
PIPELINE_SCRIPT = Path(r"C:\Users\lucas\OneDrive\Documentos\Lucas\Projeto-Tomo\codes\PROCESS-PATIENT-v5.py")
PYTHON_EXE = None  # None = usa "python" do ambiente atual. Se quiser fixar: r"C:\...\python.exe"


# =========================
# Helpers
# =========================
RESULTS_TXT_PATTERNS = [
    re.compile(r"^\s*RESULTS_TXT:\s*(.+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[RESNET\]\s*Resultados salvos em:\s*(.+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Resultados salvos em:\s*(.+)\s*$", re.IGNORECASE),
]


def extract_results_txt_from_line(line: str) -> Optional[Path]:
    for pat in RESULTS_TXT_PATTERNS:
        m = pat.match(line.strip())
        if m:
            p = m.group(1).strip().strip('"').strip("'")
            try:
                return Path(p)
            except Exception:
                return None
    return None


def parse_summary_from_txt(txt_path: Path) -> str:
    if not txt_path.exists():
        return f"(não achei o arquivo de resultado)\n{txt_path}"

    text = txt_path.read_text(encoding="utf-8", errors="replace")

    diag = None
    segs = None

    m = re.search(r"(DIAGNÓSTICO DO PACIENTE:.*)", text)
    if m:
        diag = m.group(1).strip()

    m2 = re.search(r"SEGMENTO\(S\) COM FRATURA DETECTADA NA JANELA:\s*\n(.*)", text)
    if m2:
        segs = m2.group(1).strip()

    out = []
    out.append(f"Arquivo: {txt_path}")
    out.append("")
    out.append(diag or "DIAGNÓSTICO: (não encontrado no txt)")
    out.append(segs or "SEGMENTOS: (não encontrado no txt)")
    return "\n".join(out)


def run_pipeline(folder: Path, log_cb, done_cb):
    """
    Roda o pipeline como subprocesso:
      python -u PROCESS-PATIENT-v3.py <PASTA_ESCOLHIDA>
    Captura stdout/stderr e manda para log_cb.
    """
    py = str(PYTHON_EXE) if PYTHON_EXE else "python"
    cmd = [py, "-u", str(PIPELINE_SCRIPT), str(folder)]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    results_txt_path = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        # stream ao vivo
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            log_cb(line)

            # tenta capturar caminho do TXT do output do próprio pipeline
            p = extract_results_txt_from_line(line)
            if p is not None:
                results_txt_path = p

        code = proc.wait()

        if code != 0:
            done_cb(False, f"Erro: o pipeline terminou com código {code}.", None)
            return

        done_cb(True, "Concluído!", results_txt_path)

    except Exception as e:
        done_cb(False, f"Falha ao executar: {e}", None)


# =========================
# GUI
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pipeline YOLOv7 + ResNet")
        self.geometry("980x650")

        self.folder_path: Optional[Path] = None
        self.selected_folder = tk.StringVar(value="(nenhuma pasta selecionada)")
        self.status = tk.StringVar(value="Pronto.")

        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        tk.Label(top, text="Pasta de entrada:").pack(side="left")
        tk.Entry(top, textvariable=self.selected_folder, width=95).pack(side="left", padx=8)
        tk.Button(top, text="Escolher...", command=self.choose_folder).pack(side="left")

        btns = tk.Frame(self)
        btns.pack(fill="x", padx=10)

        self.run_btn = tk.Button(btns, text="Rodar", command=self.on_run)
        self.run_btn.pack(side="left")

        tk.Button(btns, text="Limpar", command=self.clear_logs).pack(side="left", padx=8)

        tk.Label(btns, textvariable=self.status).pack(side="left", padx=12)

        self.text = tk.Text(self, wrap="word")
        self.text.pack(fill="both", expand=True, padx=10, pady=10)

        # Área de resumo (resultado final)
        bottom = tk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(bottom, text="Resumo:").pack(anchor="w")
        self.summary = tk.Text(bottom, height=6, wrap="word")
        self.summary.pack(fill="x")

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Selecione a pasta de entrada")
        if folder:
            self.folder_path = Path(folder)
            self.selected_folder.set(str(self.folder_path))

    def clear_logs(self):
        self.text.delete("1.0", "end")
        self.summary.delete("1.0", "end")

    def log(self, msg: str):
        self.text.insert("end", msg + "\n")
        self.text.see("end")

    def set_summary(self, msg: str):
        self.summary.delete("1.0", "end")
        self.summary.insert("end", msg)
        self.summary.see("end")

    def on_run(self):
        if not PIPELINE_SCRIPT.exists():
            messagebox.showerror("Erro", f"PIPELINE_SCRIPT não encontrado:\n{PIPELINE_SCRIPT}")
            return

        if self.folder_path is None or not self.folder_path.exists():
            messagebox.showwarning("Atenção", "Selecione uma pasta primeiro.")
            return

        self.run_btn.config(state="disabled")
        self.status.set("Rodando...")
        self.set_summary("")

        # roda em thread para não travar UI
        def worker():
            def log_cb(line: str):
                self.after(0, lambda: self.log(line))

            def done_cb(ok: bool, message: str, results_txt_path: Optional[Path]):

                def finish():
                    self.status.set("Concluído." if ok else "Falhou.")
                    self.run_btn.config(state="normal")

                    self.log("")
                    self.log(message)
                    self.log("")

                    # Tenta montar um resumo:
                    if ok:
                        if results_txt_path is not None and results_txt_path.exists():
                            self.set_summary(parse_summary_from_txt(results_txt_path))
                        else:
                            # Se não pegou pelo log, pelo menos avisa
                            self.set_summary(
                                "Terminei, mas não consegui detectar automaticamente o caminho do TXT.\n"
                                "Se quiser, adicione no final do pipeline:\n"
                                "  print(f\"RESULTS_TXT: {RESULTS_TXT}\", flush=True)\n"
                            )
                    else:
                        self.set_summary(message)

                self.after(0, finish)

            run_pipeline(self.folder_path, log_cb, done_cb)

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
