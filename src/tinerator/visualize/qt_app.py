import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QCoreApplication, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebEngineWidgets import QWebEngineProfile


class MainWindowWeb(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tw_title = "TINerator"
        self.browser = QWebEngineView()
        self.browser.loadFinished.connect(self.onLoadFinished)
        self.setCentralWidget(self.browser)
        self.setWindowTitle(f"{self.tw_title} (Loading...)")

    def setParams(
        self, title: str = None, window_size: tuple = None, allow_resize: bool = False
    ):
        if title:
            self.tw_title = title
            self.setWindowTitle(f"{title} (Loading...)")

        if window_size:
            self.resize(window_size[0], window_size[1])

        if not allow_resize:
            self.setFixedSize(self.width(), self.height())

    def loadURL(self, url: str):
        self.browser.load(QUrl(url))

    def onLoadFinished(self):
        self.setWindowTitle(self.tw_title)

    def closeEvent(self, event):
        self.setWindowTitle(f"{self.tw_title} (Closing...)")
        self.browser.deleteLater()
        self.browser.stop()
        self.browser.destroy()
        del self.browser
        self.close()
        QCoreApplication.quit()


def run_web_app(
    url: str,
    title: str = "TINerator",
    width: int = 900,
    height: int = 600,
    allow_resize: bool = True,
):
    qt_app = QtWidgets.QApplication.instance()

    if qt_app is None:
        qt_app = QApplication(sys.argv)

    qt_app.setQuitOnLastWindowClosed(True)

    window = MainWindowWeb()
    window.setParams(
        title=title, window_size=(width, height), allow_resize=allow_resize
    )
    window.loadURL(url)
    window.show()
    err = qt_app.exec_()

    del window
    del qt_app

    return err
