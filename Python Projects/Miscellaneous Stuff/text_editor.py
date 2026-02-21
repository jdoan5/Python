import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QAction, QFileDialog

class TextEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('Text Editor')

        openAction = QAction('&Open', self)
        openAction.triggered.connect(self.openFile)

        saveAction = QAction('&Save', self)
        saveAction.triggered.connect(self.saveFile)

        viewAction = QAction('&View', self)
        viewAction.triggered.connect(self.viewFile)

        menubar = self.menuBar()
        fileMenu1 = menubar.addMenu('&File')
        fileMenu = menubar.addMenu('&View')
        fileMenu1.addAction(openAction)
        fileMenu1.addAction(saveAction)
        fileMenu1.addAction(viewAction)

    def openFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '/')
        if filename:
            with open(filename, 'r') as file:
                self.textEdit.setText(file.read())

    def saveFile(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save File', '/')
        if filename:
            with open(filename, 'w') as file:
                file.write(self.textEdit.toPlainText())

    def viewFile(self):
        filename, _= QFileDialog.getViewFileName(self, 'View File', '/')
        if filename:
            with open(filename, 'r') as file:
                self.textEdit.setText(file.read())

def main():
    app = QApplication(sys.argv)
    editor = TextEditor()
    editor.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
