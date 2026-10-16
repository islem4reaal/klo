import os
import webview
 
from backend import data_loader
 
 
class Api:
    """Methods exposed to the frontend as window.pywebview.api.*"""
 
    def ping(self):
        return "pong from Python — bridge is working"
 
    def load_file(self):
        """
        Prompt for a file, load it via data_loader, and return a
        JSON-serializable dict the page can render directly.
 
        Success shape (whatever data_loader.load_excel returns) plus:
            {"ok": True, "filename": "sales.xlsx", ...}
        Failure shape:
            {"ok": False, "error": "human-readable message"}
        """
        window = webview.windows[0]
        selection = window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types = ("Excel Files (*.xlsx;*.xls)",
                        "CSV Files (*.csv)",
                        "All files (*.*)"
                        ),
        )
 
        if not selection:
            return {"ok": False, "error": "No file selected"}
 
        path = selection[0]
 
        try:
            data = data_loader.load_filesub(path)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            return {"ok": False, "error": str(exc)}
 
        data["ok"] = True
        data["filename"] = os.path.basename(path)
        return data
 
 
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "frontend", "index.html")
 
    webview.create_window(
        title="Klo",
        url=html_path,
        js_api=Api(),
        width=1150,
        height=700,
        min_size=(850, 500),
    )
    webview.start(debug=True)
 
 
if __name__ == "__main__":
    main()
