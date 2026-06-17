# PyInstaller spec — builds a double-click app on macOS (.app) and Windows (.exe).
#
# Build:
#   pip install pyinstaller
#   pyinstaller packaging.spec
#
# Output lands in dist/:
#   macOS   -> dist/XMLFlowVisualizer.app
#   Windows -> dist/XMLFlowVisualizer/XMLFlowVisualizer.exe
#
# The same spec works on both OSes; PyInstaller detects the platform. Run it on
# each OS you want to ship for (you cannot cross-compile a Mac app on Windows).

import sys

block_cipher = None

a = Analysis(
    ["gui_app.py"],
    pathex=[],
    binaries=[],
    datas=[("sample_eip.xml", ".")],  # bundle the sample so "Load Sample" works
    hiddenimports=["matplotlib.backends.backend_agg"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="XMLFlowVisualizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed app, no terminal
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="XMLFlowVisualizer",
)

# On macOS, also wrap the collection into a .app bundle.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="XMLFlowVisualizer.app",
        icon=None,
        bundle_identifier="com.jdoan.xmlflowvisualizer",
    )
