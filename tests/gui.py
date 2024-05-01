import sys
import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog
from PySide6.QtGui import QAction
from pyvistaqt import QtInteractor
from matplotlib.colors import LinearSegmentedColormap
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

colors = [
    [0, 0, 255], [0, 93, 255], [0, 185, 255], [0, 255, 232],
    [0, 255, 139], [0, 255, 46], [46, 255, 0], [139, 255, 0],
    [232, 255, 0], [255, 185, 0], [255, 93, 0], [255, 0, 0]
]
colors = np.array(colors) / 255.0
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=12)

class FiniteElementPostProcessor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Finite Element Post-Processor")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Create PyVista interactor
        self.interactor = QtInteractor()
        points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0]])
        cells = [[3, 0, 1, 2]]
        dargs = dict(
            scalars="PhaseField",
            show_scalar_bar=False,
            show_vertices=False,
            show_edges=False,
            vertex_color='red',
            point_size=15,
            cmap=custom_cmap,
        )
        mesh = pv.read("F:\\GitHub\\pyfem\\examples\\mechanical_phase\\CT\\Job-1-100.vtu")
        self.interactor.add_mesh(mesh, **dargs)
        # layout.addWidget(self.interactor)

        # Create QtConsole
        self.console_widget = RichJupyterWidget()
        layout.addWidget(self.console_widget)

        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel
        kernel.gui = 'qt'
        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        objects_to_inject = {'np': np, 'pv': pv}  # Add any other objects you want to inject

        for name, obj in objects_to_inject.items():
            kernel.shell.push({name: obj})

        self.create_menu()

    def create_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        open_action = QAction("Close", self)
        open_action.triggered.connect(self.close)
        file_menu.addAction(open_action)

    def open_file(self):
        self.interactor.clear()

        file_path, _ = QFileDialog.getOpenFileName(self, "Open VTU File", "", "VTU Files (*.vtu)")
        if file_path:
            self.load_vtu_file(file_path)
            self.load_vtu_file_volume_rendering(file_path)

    def close(self):
        self.interactor.clear()

    def load_vtu_file(self, file_path):
        mesh = pv.read(file_path)
        self.interactor.add_mesh(mesh, show_edges=True)

    def load_vtu_file_volume_rendering(self, file_path):
        mesh = pv.read(file_path)
        self.interactor.add_volume(mesh, cmap="coolwarm", opacity="linear")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FiniteElementPostProcessor()
    window.show()
    sys.exit(app.exec())
