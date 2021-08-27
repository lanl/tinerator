def init_window(dash_app, server_args=(), url: str = "http://127.0.0.1:8050/"):
    '''
    Local renderer for plots.
    '''
    import multiprocessing
    from PyQt5.QtCore import QUrl
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtWidgets import QApplication

    qt_app = QApplication(['TINerator',])
    webview = QWebEngineView()

    #server = multiprocessing.Process(target=dash_app.run_server, args=server_args)
    import threading
    server = threading.Thread(target=lambda: dash_app.run_server())#host=host_name, port=port, debug=True, use_reloader=False)).start()

    webview.load(QUrl(url))
    webview.show()
    server.start()
    qt_app.exec_()
    server.join()
    #server.terminate()