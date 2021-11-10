import subprocess
import sys
import os


def run_web_app(
    url: str, width: int = 800, height: int = 600, min_size: tuple = (800, 600)
):
    cmd_py = ["import webview"]
    cmd_py += [
        f"window = webview.create_window("
        f"'TINerator', url='{url}', width={width}, "
        f"height={height}, min_size={min_size})"
    ]
    cmd_py += ["window.closing += window.destroy"]
    cmd_py += ["webview.start()"]

    # TODO: on quit, this should kill the server and return
    # so, use Popen with wait() while server.isAlive
    cmd_sh = [sys.executable, "-c", '"' + ";".join(cmd_py) + '"']
    exit_code = subprocess.call(" ".join(cmd_sh), cwd=os.getcwd(), shell=True)
    return exit_code
