"""
This is not a unit test; rather, it is a general script
to validate whether or not PyQt5 and QWebEngineView are installed
and working on your system. These are the preferred non-Jupyter backends
for plotting in TINerator.
"""
import sys

def testWebEngine(url = "https://www.github.com/lanl/tinerator"):
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QUrl
    from PyQt5.QtWebEngineWidgets import QWebEngineView

    app = QApplication(sys.argv)
    browser = QWebEngineView()
    browser.load(QUrl(url))
    browser.show()

    exit_code = app.exec_()
    return exit_code

#url = "https://www.github.com/lanl/tinerator"
url = "http://127.0.0.1:8050/"

exit_code = testWebEngine(url = url)
print(f"Browser window closed. Test successful. {exit_code=}")