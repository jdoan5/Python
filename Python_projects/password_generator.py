import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel
import random
import string

class PasswordGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Password Generator')
        self.layout = QVBoxLayout()

        self.label = QLabel('Generated Password:')
        self.layout.addWidget(self.label)

        self.passwordOutput = QLineEdit()
        self.passwordOutput.setReadOnly(True)
        self.layout.addWidget(self.passwordOutput)

        self.generateButton = QPushButton('Generate Password')
        self.generateButton.clicked.connect(self.generatePassword)
        self.layout.addWidget(self.generateButton)

        self.setLayout(self.layout)
        self.resize(400, 200)

    def generatePassword(self):
        length = 12  # Define the length of the password
        characters = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(random.choice(characters) for i in range(length))
        self.passwordOutput.setText(password)

def main():
    app = QApplication(sys.argv)
    ex = PasswordGenerator()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
