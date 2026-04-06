import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.theme import DARK_THEME


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_THEME)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
