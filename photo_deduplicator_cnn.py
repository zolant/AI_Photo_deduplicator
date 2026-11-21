"""Find visually similar photos with a CNN (ONNX Runtime) and review/delete them in a GUI.

CNN version of photo_deduplicator.py. Instead of a 64-bit perceptual hash it
compares 1280-number feature vectors from a pretrained image network, so it
also finds copies that are padded onto a canvas, cropped, filtered or
re-encoded, and near-identical shots of the same scene.
Selected photos can be moved to Trash or to a "deleted" folder (suggested next to
the scanned folder, editable), preserving their paths relative to the scanned folder.

Models (downloaded once from the official ONNX Model Zoo on GitHub, or put the
.onnx file in a "models" folder next to this script for offline use):
    EfficientNet-Lite4  52 MB  most accurate (default)
    MobileNetV2         14 MB  about 3x faster, slightly less accurate

Acceleration is detected automatically:
    Windows  DirectML GPU (needs onnxruntime-directml; suggested when a GPU is found)
    macOS    Core ML (Apple GPU / Neural Engine, included in onnxruntime)
    else     CPU

Requires: numpy, Pillow, onnxruntime, onnx (onnx is only used once per model to
prepare it). Optional: send2trash, pillow-heif, truststore.

Run:
    Windows (no console window):  pythonw photo_deduplicator_cnn.py
    macOS / Linux (e.g. Mint):    python3 photo_deduplicator_cnn.py
"""
import hashlib
import importlib.util
import logging
import os
import platform
import queue
import shutil
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk

# Local photos are trusted; Pillow's default limit (~179 MP) would silently skip
# large panoramas and upscaled copies.
Image.MAX_IMAGE_PIXELS = None

try:
    # Use the OS certificate store so downloads work behind corporate TLS proxies.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

OS_NAME = platform.system()
IS_WINDOWS = OS_NAME == 'Windows'
IS_MACOS = OS_NAME == 'Darwin'

if IS_WINDOWS:
    UI_FONT_FAMILY = 'Segoe UI'
    TRASH_NAME = 'Recycle Bin'
    MODEL_CACHE_DIR = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
                                   'PhotoDeduplicator', 'models')
elif IS_MACOS:
    UI_FONT_FAMILY = 'Helvetica Neue'
    TRASH_NAME = 'Trash'
    MODEL_CACHE_DIR = os.path.expanduser('~/Library/Caches/PhotoDeduplicator/models')
else:
    UI_FONT_FAMILY = 'DejaVu Sans'
    TRASH_NAME = 'Trash'
    MODEL_CACHE_DIR = os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
                                   'photo-deduplicator', 'models')
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.jpe', '.jfif', '.gif', '.bmp', '.tif', '.tiff',
                    '.png', '.webp', '.avif',
                    # Readable only with pillow-heif; otherwise reported as unreadable.
                    '.heic', '.heif'}
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Folders never scanned: deleting files inside a Photos library corrupts it.
SKIP_DIR_SUFFIXES = ('.photoslibrary', '.photolibrary', '.aplibrary')
THUMB_SIZE = (640, 640)
GRID_COLUMNS = 3
DECODE_WORKERS = min(8, max(1, os.cpu_count() or 1))
DECODE_CHUNK = 32
APP_TITLE = 'Photo Deduplicator (CNN)'
MODEL_INPUT_SIZE = 224

_ZOO = 'https://github.com/onnx/models/raw/main/validated/vision/classification'
MODELS = {
    'efficientnet': {
        'label': 'EfficientNet-Lite4 (52 MB, most accurate)',
        'file': 'efficientnet-lite4-11.onnx',
        'url': f'{_ZOO}/efficientnet-lite4/model/efficientnet-lite4-11.onnx',
        'sha256': 'd111689907c06eea7c82e4833ddef758da6453b9d4cf60b7e99ca05c7cbd9c12',
        'featureOutput': 'efficientnet-lite4/model/head/Squeeze:0',
        'channelsFirst': False,
        'mean': np.array([127.0, 127.0, 127.0], dtype=np.float32),
        'std': np.array([128.0, 128.0, 128.0], dtype=np.float32),
        'batched': False,
        'defaultThreshold': 0.95,
    },
    'mobilenet': {
        'label': 'MobileNetV2 (14 MB, fastest)',
        'file': 'mobilenetv2-12.onnx',
        'url': f'{_ZOO}/mobilenet/model/mobilenetv2-12.onnx',
        'sha256': 'c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5',
        'featureOutput': '472',
        'channelsFirst': True,
        'mean': np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255,
        'std': np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255,
        'batched': True,
        'defaultThreshold': 0.96,
    },
}
DEFAULT_MODEL = 'efficientnet'

PROVIDER_NAMES = {
    'CUDAExecutionProvider': 'NVIDIA CUDA GPU',
    'DmlExecutionProvider': 'DirectML GPU',
    'CoreMLExecutionProvider': 'Core ML (Apple GPU / Neural Engine)',
    'CPUExecutionProvider': 'CPU',
}
DIRECTML_INSTALL_CMD = 'py -m pip uninstall -y onnxruntime && py -m pip install onnxruntime-directml'


class ModelUnavailableError(Exception):
    """The model could not be downloaded; the user can pick a local copy instead."""

    def __init__(self, modelKey, reason):
        super().__init__(reason)
        self.modelKey = modelKey


# ---------------------------------------------------------------- acceleration

def availableProviders():
    import onnxruntime as ort
    available = ort.get_available_providers()
    return [p for p in PROVIDER_NAMES if p in available]


def detectWindowsGpus():
    """Names of real display adapters (empty on non-Windows or on failure)."""
    if not IS_WINDOWS:
        return []
    try:
        result = subprocess.run(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command',
             '(Get-CimInstance Win32_VideoController).Name'],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception:
        return []
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [n for n in names if not n.lower().startswith(('microsoft basic', 'microsoft remote'))]


def createSession(modelPath):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # errors only; Core ML partition notices are noise
    providers = availableProviders()
    options = ort.SessionOptions()
    if 'DmlExecutionProvider' in providers:
        # Required by DirectML.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        return ort.InferenceSession(modelPath, options, providers=providers)
    except Exception as exc:
        logging.warning('Accelerated session failed (%s); falling back to CPU', exc)
        return ort.InferenceSession(modelPath, ort.SessionOptions(),
                                    providers=['CPUExecutionProvider'])


# ---------------------------------------------------------------------- models

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _featureModelPath(modelKey):
    return os.path.join(MODEL_CACHE_DIR, MODELS[modelKey]['file'].replace('.onnx', '-features.onnx'))


def _downloadModel(modelKey, onStatus):
    spec = MODELS[modelKey]
    target = os.path.join(MODEL_CACHE_DIR, spec['file'])
    partial = target + '.part'
    try:
        request = urllib.request.Request(spec['url'], headers={'User-Agent': 'PhotoDeduplicator'})
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, 'wb') as out:
            total = int(response.headers.get('Content-Length') or 0)
            done = 0
            while True:
                block = response.read(1 << 18)
                if not block:
                    break
                out.write(block)
                done += len(block)
                onStatus(f'Downloading model: {done / 1e6:.1f} of {total / 1e6:.1f} MB'
                         if total else f'Downloading model: {done / 1e6:.1f} MB')
    except Exception as exc:
        if os.path.exists(partial):
            os.remove(partial)
        raise ModelUnavailableError(modelKey, f'Download failed: {exc}') from exc
    if _sha256(partial) != spec['sha256']:
        os.remove(partial)
        # Typically a proxy/firewall page returned instead of the model.
        raise ModelUnavailableError(modelKey, 'Downloaded file is not the expected model '
                                              '(blocked by a proxy or firewall?)')
    os.replace(partial, target)
    return target


def _prepareFeatureModel(modelKey, sourcePath):
    """Save a copy of the model whose output is the pooled feature vector, not class scores."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError('The "onnx" package is needed once to prepare the model.\n'
                           'Install it with:  pip install onnx') from exc
    spec = MODELS[modelKey]
    model = onnx.load(sourcePath)
    producedNames = {name for node in model.graph.node for name in node.output}
    if spec['featureOutput'] not in producedNames:
        raise RuntimeError(f'{os.path.basename(sourcePath)} is not the expected '
                           f'{spec["label"]} model (feature layer not found).')
    del model.graph.output[:]
    # Typed output: Core ML rejects untyped graph outputs.
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(spec['featureOutput'], onnx.TensorProto.FLOAT, None))
    # Drop the now-unused classifier nodes; Core ML fails on dangling partitions.
    needed = {spec['featureOutput']}
    keptNodes = []
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            keptNodes.append(node)
            needed.update(name for name in node.input if name)
    del model.graph.node[:]
    model.graph.node.extend(reversed(keptNodes))
    keptInitializers = [init for init in model.graph.initializer if init.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keptInitializers)
    target = _featureModelPath(modelKey)
    onnx.save(model, target + '.part')
    os.replace(target + '.part', target)
    return target


def ensureModel(modelKey, onStatus, manualPath=None):
    """Return the path of the ready-to-use feature model, downloading/preparing it if needed."""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    featurePath = _featureModelPath(modelKey)
    if os.path.exists(featurePath):
        return featurePath
    spec = MODELS[modelKey]
    sourcePath = manualPath
    if sourcePath is None:
        for candidate in (os.path.join(LOCAL_MODEL_DIR, spec['file']),
                          os.path.join(MODEL_CACHE_DIR, spec['file'])):
            if os.path.exists(candidate):
                sourcePath = candidate
                break
    if sourcePath is None:
        sourcePath = _downloadModel(modelKey, onStatus)
    onStatus('Preparing model (one-time)...')
    return _prepareFeatureModel(modelKey, sourcePath)


# ------------------------------------------------------------------- embedding

def loadModelPixels(imagePath):
    """Whole photo (no crop) as a 224x224 uint8 array, EXIF-rotated, transparency on white.

    Converted to greyscale (replicated to 3 channels) so black-and-white or
    colour-adjusted copies match their colour originals.
    """
    with Image.open(imagePath) as im:
        # Fast JPEG downscale on load; no-op for other formats.
        im.draft('RGB', (MODEL_INPUT_SIZE * 2, MODEL_INPUT_SIZE * 2))
        im = ImageOps.exif_transpose(im)
        if im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in im.info:
            rgba = im.convert('RGBA')
            im = Image.new('RGBA', rgba.size, 'white')
            im.alpha_composite(rgba)
        im = im.convert('L').resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                    Image.Resampling.BICUBIC, reducing_gap=3.0)
        return np.repeat(np.asarray(im)[:, :, None], 3, axis=2)


def _loadVariants(imagePath, matchRotations):
    try:
        pixels = loadModelPixels(imagePath)
    except Exception as exc:
        logging.warning('Cannot read %s: %s', imagePath, exc)
        return imagePath, None, str(exc)
    turns = range(4) if matchRotations else range(1)
    return imagePath, [np.rot90(pixels, k) for k in turns], None


class Embedder:
    def __init__(self, modelKey, modelPath):
        self.spec = MODELS[modelKey]
        self.session = createSession(modelPath)
        self.inputName = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]

    def embed(self, images):
        """L2-normalised feature vectors for a list of HxWx3 uint8 images."""
        batch = (np.stack(images).astype(np.float32) - self.spec['mean']) / self.spec['std']
        if self.spec['channelsFirst']:
            batch = batch.transpose(0, 3, 1, 2)
        batch = np.ascontiguousarray(batch)
        if self.spec['batched']:
            features = self.session.run(None, {self.inputName: batch})[0].reshape(len(images), -1)
        else:
            features = np.stack([self.session.run(None, {self.inputName: batch[i:i + 1]})[0].reshape(-1)
                                 for i in range(len(images))])
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-12)


def computeEmbeddings(imagePaths, embedder, matchRotations, onProgress=None):
    """Return (paths, embeddings[N, V, D], {unreadablePath: reason})."""
    paths, vectors, unreadable = [], [], {}
    done = 0
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        for start in range(0, len(imagePaths), DECODE_CHUNK):
            chunk = imagePaths[start:start + DECODE_CHUNK]
            for path, variants, error in pool.map(
                    lambda p: _loadVariants(p, matchRotations), chunk):
                done += 1
                if variants is None:
                    unreadable[path] = error
                else:
                    paths.append(path)
                    vectors.append(embedder.embed(variants))
                if onProgress:
                    onProgress(done, len(imagePaths))
    embeddings = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, 1, 1), np.float32)
    return paths, embeddings, unreadable


def similarityMatrixRows(embeddings, rowStart, rowEnd):
    """Cosine similarity of rows [rowStart, rowEnd) against all photos, best over rotations."""
    count, variants, dims = embeddings.shape
    base = embeddings[:, 0, :]
    allVariants = embeddings.reshape(count * variants, dims)
    forward = (base[rowStart:rowEnd] @ allVariants.T).reshape(rowEnd - rowStart, count, variants).max(axis=2)
    backward = (allVariants.reshape(count, variants, dims)[rowStart:rowEnd].reshape(-1, dims)
                @ base.T).reshape(rowEnd - rowStart, variants, count).max(axis=1)
    return np.maximum(forward, backward)


def findDuplicates(paths, embeddings, threshold, chunkRows=256):
    """Map each image to the other images with similarity >= threshold."""
    duplicates = {p: [] for p in paths}
    for rowStart in range(0, len(paths), chunkRows):
        rowEnd = min(rowStart + chunkRows, len(paths))
        sims = similarityMatrixRows(embeddings, rowStart, rowEnd)
        for offset, row in enumerate(sims):
            i = rowStart + offset
            for j in np.nonzero(row[i + 1:] >= threshold)[0] + i + 1:
                duplicates[paths[i]].append(paths[j])
                duplicates[paths[j]].append(paths[i])
    return duplicates


# ----------------------------------------------------------------- file helpers

def imgRes(imagePath):
    try:
        with Image.open(imagePath) as im:
            return im.size
    except Exception:
        return 0, 0


def fileSize(imagePath):
    try:
        return os.path.getsize(imagePath)
    except OSError:
        return 0


def collectImages(srcDir):
    imagePaths = []
    for root, dirs, files in os.walk(srcDir):
        dirs[:] = [d for d in dirs if not d.startswith('.')
                   and not d.lower().endswith(SKIP_DIR_SUFFIXES)
                   and d not in ('$RECYCLE.BIN', 'System Volume Information')]
        for fileName in files:
            # '._*' are macOS resource-fork files left on non-Mac drives.
            if fileName.startswith('._'):
                continue
            if os.path.splitext(fileName)[1].lower() in IMAGE_EXTENSIONS:
                imagePaths.append(os.path.normpath(os.path.join(root, fileName)))
    return imagePaths


def buildGroups(duplicates):
    """Turn the duplicate map into unique groups sorted by file size (largest first)."""
    seen = set()
    groups = []
    for key, dupList in duplicates.items():
        if not dupList or key in seen:
            continue
        group = [p for p in [key, *dupList] if p not in seen]
        seen.update(group)
        if len(group) > 1:
            group.sort(key=fileSize, reverse=True)
            groups.append(group)
    return groups


def findSimilarGroups(srcDir, modelKey, threshold, matchRotations,
                      onProgress=None, onStatus=None, manualModelPath=None):
    """Return (groups, {unreadablePath: reason}, similarityFn, providerName)."""
    onStatus = onStatus or (lambda _text: None)
    modelPath = ensureModel(modelKey, onStatus, manualModelPath)
    onStatus('Loading model...')
    embedder = Embedder(modelKey, modelPath)
    providerName = PROVIDER_NAMES.get(embedder.provider, embedder.provider)
    onStatus(f'Collecting images... (running on {providerName})')
    imagePaths = collectImages(srcDir)
    if onProgress:
        onProgress(0, len(imagePaths))
    paths, embeddings, unreadable = computeEmbeddings(imagePaths, embedder, matchRotations, onProgress)
    if not paths:
        return [], unreadable, None, providerName
    groups = buildGroups(findDuplicates(paths, embeddings, threshold))
    index = {p: i for i, p in enumerate(paths)}

    def similarity(pathA, pathB):
        i, j = index[pathA], index[pathB]
        return float(similarityMatrixRows(embeddings, i, i + 1)[0, j])

    return groups, unreadable, similarity, providerName


def _windowsRecycle(path):
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        # shellapi.h packs this struct to 1 byte on 32-bit Windows.
        _pack_ = 1 if ctypes.sizeof(ctypes.c_void_p) == 4 else 8
        _fields_ = [('hwnd', wintypes.HWND), ('wFunc', wintypes.UINT),
                    ('pFrom', wintypes.LPCWSTR), ('pTo', wintypes.LPCWSTR),
                    ('fFlags', ctypes.c_uint16), ('fAnyOperationsAborted', wintypes.BOOL),
                    ('hNameMappings', ctypes.c_void_p), ('lpszProgressTitle', wintypes.LPCWSTR)]

    foDelete = 0x0003
    flags = 0x0040 | 0x0010 | 0x0004 | 0x0400  # ALLOWUNDO|NOCONFIRMATION|SILENT|NOERRORUI
    # pFrom must be double-NUL terminated; ctypes adds the second NUL.
    op = SHFILEOPSTRUCTW(wFunc=foDelete, pFrom=os.path.abspath(path) + '\0', fFlags=flags)
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if result != 0 or op.fAnyOperationsAborted:
        raise OSError(f'Recycle Bin operation failed (code {result:#x})')


def _macTrash(path):
    # Finder keeps "Put Back" information; the path is passed as an argument, not inlined.
    script = ['on run argv',
              'tell application "Finder" to delete (POSIX file (item 1 of argv) as alias)',
              'end run']
    cmd = ['osascript'] + [arg for line in script for arg in ('-e', line)] + [os.path.abspath(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or 'Finder could not move the file to Trash')


def deleteFile(path):
    """Move a file to the Recycle Bin (Windows) or Trash (macOS/Linux)."""
    try:
        from send2trash import send2trash
        send2trash(path)
        return
    except ImportError:
        pass
    if IS_WINDOWS:
        _windowsRecycle(path)
    elif IS_MACOS:
        _macTrash(path)
    else:
        raise RuntimeError('Install Send2Trash to move files to the Trash on this system.')


def _isWithin(path, directory):
    path = os.path.normcase(os.path.realpath(path))
    directory = os.path.normcase(os.path.realpath(directory))
    try:
        return os.path.commonpath((path, directory)) == directory
    except ValueError:
        return False


def defaultDeletedFolder(srcDir):
    """Suggested deleted folder: next to the scanned folder."""
    return os.path.join(os.path.dirname(os.path.abspath(srcDir)), 'deleted')


def validateDeletedFolder(path, srcDir):
    """Return (absolutePath, None) when the folder is usable, otherwise (None, reason)."""
    path = path.strip()
    if not path:
        return None, 'Enter a folder for the deleted photos.'
    path = os.path.abspath(os.path.expanduser(path))
    if _isWithin(path, srcDir):
        return None, 'The deleted folder must be outside the scanned folder.'
    if os.path.exists(path) and not os.path.isdir(path):
        return None, f'A file with this name already exists:\n{path}'
    if not os.path.isdir(os.path.dirname(path)):
        return None, f'The parent folder does not exist:\n{os.path.dirname(path)}'
    return path, None


def chooseDeletedFolder(root, srcDir):
    """Show the deleted-folder path in an editable text box; return the path or None if cancelled."""
    dialog = tk.Toplevel(root)
    dialog.title('Deleted folder')
    dialog.transient(root)
    dialog.resizable(True, False)
    dialog.geometry(f'+{root.winfo_rootx() + 40}+{root.winfo_rooty() + 40}')

    frame = tk.Frame(dialog, padx=15, pady=15)
    frame.pack(fill='both', expand=True)
    tk.Label(frame, anchor='w', justify='left', wraplength=560,
             text='Selected photos will be moved to this folder. Their sub-folders '
                  f'relative to the scanned folder are kept:\n{srcDir}').pack(fill='x')

    pathVar = tk.StringVar(value=defaultDeletedFolder(srcDir))
    row = tk.Frame(frame)
    row.pack(fill='x', pady=10)
    entry = tk.Entry(row, textvariable=pathVar, width=64)
    entry.pack(side='left', fill='x', expand=True)
    result = {'path': None}

    def browse():
        current = os.path.abspath(os.path.expanduser(pathVar.get().strip()
                                                      or defaultDeletedFolder(srcDir)))
        startDir = current if os.path.isdir(current) else os.path.dirname(current)
        picked = filedialog.askdirectory(
            title='Select the parent folder (a "deleted" folder is created in it)',
            initialdir=startDir, parent=dialog)
        if picked:
            picked = os.path.normpath(picked)
            isDeletedFolder = os.path.basename(picked).lower() == 'deleted'
            pathVar.set(picked if isDeletedFolder else os.path.join(picked, 'deleted'))
        dialog.lift()
        entry.focus_set()

    def accept():
        path, error = validateDeletedFolder(pathVar.get(), srcDir)
        if error:
            messagebox.showerror('Invalid deleted folder', error, parent=dialog)
            entry.focus_set()
            return
        result['path'] = path
        dialog.destroy()

    tk.Button(row, text='Browse…', command=browse).pack(side='left', padx=(6, 0))
    buttons = tk.Frame(frame)
    buttons.pack(fill='x')
    tk.Button(buttons, text='Cancel', width=10, command=dialog.destroy).pack(side='right')
    tk.Button(buttons, text='OK', width=10, default='active',
              command=accept).pack(side='right', padx=6)
    dialog.bind('<Return>', lambda _e: accept())
    dialog.bind('<Escape>', lambda _e: dialog.destroy())
    dialog.protocol('WM_DELETE_WINDOW', dialog.destroy)

    def grabWhenVisible(attemptsLeft=40):
        # On Linux/X11 a grab fails until the window manager has mapped the window.
        if not dialog.winfo_exists():
            return
        try:
            dialog.grab_set()
        except tk.TclError:
            if attemptsLeft:
                dialog.after(50, grabWhenVisible, attemptsLeft - 1)

    dialog.update_idletasks()
    grabWhenVisible()
    dialog.lift()
    entry.focus_set()
    entry.icursor('end')
    dialog.wait_window()
    return result['path']


def moveToDeletedFolder(path, srcDir, destination):
    """Move a photo into the deleted folder, keeping its path relative to the scanned folder."""
    if not _isWithin(path, srcDir) or _isWithin(destination, srcDir):
        raise ValueError('The source or deleted folder is outside its permitted location.')
    relativePath = os.path.relpath(os.path.abspath(path), os.path.abspath(srcDir))
    if relativePath == os.pardir or relativePath.startswith(os.pardir + os.sep):
        raise ValueError('The photo is outside the scanned folder.')
    targetDir = os.path.join(destination, os.path.dirname(relativePath))
    os.makedirs(targetDir, exist_ok=True)
    if not _isWithin(targetDir, destination):
        raise ValueError('The destination folder resolves outside the deleted folder.')
    filename = os.path.basename(relativePath)
    stem, extension = os.path.splitext(filename)
    number = 0
    while True:
        # Exclusive create: an existing file is never overwritten, the name gets a number.
        name = filename if number == 0 else f'{stem} ({number}){extension}'
        target = os.path.join(targetDir, name)
        try:
            targetFile = open(target, 'xb')
            break
        except FileExistsError:
            number += 1
    copied = False
    try:
        with targetFile, open(path, 'rb') as sourceFile:
            shutil.copyfileobj(sourceFile, targetFile)
        shutil.copystat(path, target)
        copied = True
    finally:
        if not copied:
            os.remove(target)
    os.remove(path)
    return target


# -------------------------------------------------------------------------- GUI

class ScanWindow:
    """Shows model download and scan progress while the work runs in a background thread."""

    def __init__(self, root, srcDir, onDone, modelKey, threshold, matchRotations,
                 manualModelPath=None):
        self.root = root
        self.onDone = onDone
        self.messages = queue.Queue()

        self.frame = tk.Frame(root, padx=20, pady=20)
        self.frame.pack(fill='both', expand=True)
        tk.Label(self.frame, text='Searching for similar photos in:',
                 font=(UI_FONT_FAMILY, 11, 'bold'), anchor='w').pack(fill='x')
        tk.Label(self.frame, text=srcDir, anchor='w', wraplength=520,
                 justify='left').pack(fill='x', pady=(0, 10))
        self.statusLabel = tk.Label(self.frame, text='Preparing...', anchor='w')
        self.statusLabel.pack(fill='x')
        self.progress = ttk.Progressbar(self.frame, mode='indeterminate', length=520)
        self.progress.pack(fill='x', pady=10)
        self.progress.start(10)
        self.providerName = ''

        args = (srcDir, modelKey, threshold, matchRotations, manualModelPath)
        threading.Thread(target=self._worker, args=args, daemon=True).start()
        self.root.after(100, self._poll)

    def _worker(self, srcDir, modelKey, threshold, matchRotations, manualModelPath):
        try:
            result = findSimilarGroups(
                srcDir, modelKey, threshold, matchRotations,
                onProgress=lambda done, total: self.messages.put(('progress', done, total)),
                onStatus=lambda text: self.messages.put(('status', text)),
                manualModelPath=manualModelPath)
            self.messages.put(('done', result))
        except ModelUnavailableError as exc:
            self.messages.put(('model-missing', exc))
        except Exception as exc:
            logging.exception('Scan failed')
            self.messages.put(('error', exc))

    def _poll(self):
        try:
            while True:
                msg = self.messages.get_nowait()
                if msg[0] == 'status':
                    self.statusLabel.config(text=msg[1])
                    if 'running on' in msg[1]:
                        self.providerName = msg[1].split('running on ', 1)[1].rstrip(')')
                elif msg[0] == 'progress':
                    _, done, total = msg
                    if self.progress['mode'] != 'determinate':
                        self.progress.stop()
                        self.progress.config(mode='determinate', maximum=max(total, 1))
                    self.progress['value'] = done
                    self.statusLabel.config(
                        text=f'Analysing images: {done} of {total}  (running on {self.providerName})')
                else:
                    self.frame.destroy()
                    self.onDone(msg[0], msg[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


class DuplicateReviewer:
    def __init__(self, root, groups, onFinish, srcDir, deletedFolder, similarity=None):
        self.root = root
        self.groups = groups
        self.onFinish = onFinish
        self.srcDir = srcDir
        self.deletedFolder = deletedFolder
        self.similarity = similarity
        self.groupIndex = 0
        self.deletedCount = 0
        self.thumbRefs = []
        self.checkVars = []

        self.frame = tk.Frame(root)
        self.frame.pack(fill='both', expand=True)

        self.headerLabel = tk.Label(self.frame, font=(UI_FONT_FAMILY, 12, 'bold'), anchor='w')
        self.headerLabel.pack(fill='x', padx=10, pady=(10, 0))
        tk.Label(self.frame, anchor='w',
                 text='Tick the photos to delete (or click a photo). If nothing is ticked, '
                      'the group is skipped. Similarity is compared with photo 1.'
                 ).pack(fill='x', padx=10)

        container = tk.Frame(self.frame)
        container.pack(fill='both', expand=True, padx=10, pady=10)
        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.gridFrame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.gridFrame, anchor='nw')
        self.gridFrame.bind('<Configure>', lambda _e: self.canvas.configure(
            scrollregion=self.canvas.bbox('all')))
        if IS_WINDOWS or IS_MACOS:
            self.root.bind_all('<MouseWheel>', self._onMouseWheel)
        else:
            self.root.bind_all('<Button-4>', lambda _e: self.canvas.yview_scroll(-1, 'units'))
            self.root.bind_all('<Button-5>', lambda _e: self.canvas.yview_scroll(1, 'units'))

        buttons = tk.Frame(self.frame)
        buttons.pack(fill='x', padx=10, pady=(0, 10))
        tk.Button(buttons, text='Delete selected & Next', width=22,
                  command=self.deleteAndNext).pack(side='left')
        tk.Button(buttons, text='Skip (Next)', width=14,
                  command=self.nextGroup).pack(side='left', padx=8)
        tk.Button(buttons, text='Stop review', width=12,
                  command=self.finish).pack(side='right')
        self.root.bind('<Return>', lambda _e: self.deleteAndNext())

        self.showGroup()

    def _onMouseWheel(self, event):
        # Windows reports multiples of 120 per notch; macOS reports small raw deltas.
        step = -int(event.delta / 120) if IS_WINDOWS else -event.delta
        if step:
            self.canvas.yview_scroll(step, 'units')

    def showGroup(self):
        for child in self.gridFrame.winfo_children():
            child.destroy()
        self.thumbRefs.clear()
        self.checkVars.clear()
        self.canvas.yview_moveto(0)

        if self.groupIndex >= len(self.groups):
            self.finish()
            return

        group = self.groups[self.groupIndex]
        self.headerLabel.config(
            text=f'Group {self.groupIndex + 1} of {len(self.groups)} — {len(group)} similar photos')

        for idx, path in enumerate(group):
            cell = tk.Frame(self.gridFrame, bd=1, relief='solid', padx=5, pady=5)
            cell.grid(row=idx // GRID_COLUMNS, column=idx % GRID_COLUMNS, padx=5, pady=5, sticky='n')
            try:
                with Image.open(path) as im:
                    im.thumbnail(THUMB_SIZE)
                    photo = ImageTk.PhotoImage(im.convert('RGB'))
                self.thumbRefs.append(photo)
                imageLabel = tk.Label(cell, image=photo, cursor='hand2')
            except Exception as exc:
                imageLabel = tk.Label(cell, text=f'Cannot open image\n{exc}',
                                      width=40, height=15)
            imageLabel.pack()

            width, height = imgRes(path)
            if idx == 0:
                note = '  (largest)'
            elif self.similarity:
                note = f'  similarity {self.similarity(group[0], path):.0%}'
            else:
                note = ''
            checkVar = tk.BooleanVar(value=False)
            self.checkVars.append((checkVar, path))
            imageLabel.bind('<Button-1>', lambda _e, v=checkVar: v.set(not v.get()))
            tk.Checkbutton(cell, variable=checkVar, wraplength=THUMB_SIZE[0], justify='left',
                           text=f'{idx + 1}. {os.path.basename(path)}\n'
                                f'{width}x{height}, {fileSize(path) / 1024:.0f} KB{note}'
                           ).pack(anchor='w')
            tk.Label(cell, text=os.path.dirname(path), fg='grey',
                     wraplength=THUMB_SIZE[0], justify='left').pack(anchor='w')

    def deleteAndNext(self):
        selected = [path for var, path in self.checkVars if var.get()]
        if not selected:
            self.nextGroup()
            return
        if len(selected) == len(self.checkVars) and not messagebox.askyesno(
                'Confirm', 'All photos in this group are selected. Delete them all?',
                parent=self.root):
            return
        destination = self.deletedFolder or f'the {TRASH_NAME}'
        if not messagebox.askyesno(
                'Confirm delete', f'Move these files to {destination}?\n\n' + '\n'.join(selected),
                parent=self.root):
            return
        for path in selected:
            try:
                if self.deletedFolder:
                    moveToDeletedFolder(path, self.srcDir, self.deletedFolder)
                else:
                    deleteFile(path)
                self.deletedCount += 1
                logging.info('Removed: %s', path)
            except Exception as exc:
                messagebox.showerror('Delete failed', f'{path}\n{exc}', parent=self.root)
        self.nextGroup()

    def nextGroup(self):
        self.groupIndex += 1
        self.showGroup()

    def finish(self):
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.root.unbind_all(sequence)
        self.root.unbind('<Return>')
        self.frame.destroy()
        self.onFinish(self.deletedCount)


def _enableWindowsDpiAwareness():
    # Without this, Windows bitmap-scales Tk on high-DPI displays, making it blurry.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class DedupeApp:
    def __init__(self):
        if IS_WINDOWS:
            _enableWindowsDpiAwareness()
        self.root = tk.Tk()
        self.root.title(f'{APP_TITLE} ({OS_NAME})')
        self.startFrame = None
        self.lastFolder = None
        self.deletedFolder = None
        self.destinationVar = tk.StringVar(value='trash')
        self.modelVar = tk.StringVar(value=DEFAULT_MODEL)
        self.thresholdVar = tk.DoubleVar(value=MODELS[DEFAULT_MODEL]['defaultThreshold'])
        self.rotationsVar = tk.BooleanVar(value=True)
        self.gpuNames = None

    def _bringToFront(self):
        # Only lift/focus: toggling '-topmost' off on macOS sends the window behind other apps.
        self.root.lift()
        self.root.focus_force()

    def run(self):
        if importlib.util.find_spec('onnxruntime') is None:
            self.root.withdraw()
            messagebox.showerror(APP_TITLE, 'onnxruntime is not installed.\n\nInstall it with:\n'
                                 'pip install -r requirements-cnn.txt')
            self.root.destroy()
            return
        if IS_WINDOWS and 'DmlExecutionProvider' not in availableProviders():
            threading.Thread(target=self._detectGpu, daemon=True).start()
        self.showStart()
        self.root.after(200, self._bringToFront)
        self.root.mainloop()

    def _detectGpu(self):
        self.gpuNames = detectWindowsGpus()
        if self.gpuNames:
            self.root.after(0, self._refreshStartIfShown)

    def _refreshStartIfShown(self):
        if self.startFrame is not None and self.startFrame.winfo_exists():
            self.startFrame.destroy()
            self.showStart()

    def _acceleratorText(self):
        providers = availableProviders()
        best = PROVIDER_NAMES[providers[0]] if providers else 'CPU'
        return f'Acceleration: {best}'

    def _copyDirectmlCommand(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(DIRECTML_INSTALL_CMD)
        messagebox.showinfo(APP_TITLE, 'Command copied. Close this app, paste it into a '
                            'Command Prompt, then start the app again:\n\n' + DIRECTML_INSTALL_CMD,
                            parent=self.root)

    def _onModelChange(self):
        self.thresholdVar.set(MODELS[self.modelVar.get()]['defaultThreshold'])

    def showStart(self, statusText=''):
        # The folder picker is only opened from a button click in this visible window;
        # on macOS a dialog with no visible parent window loses focus and vanishes when clicked.
        self.root.geometry('640x580')
        self.startFrame = tk.Frame(self.root, padx=20, pady=20)
        self.startFrame.pack(fill='both', expand=True)
        tk.Label(self.startFrame, text='Find and remove similar photos (CNN)',
                 font=(UI_FONT_FAMILY, 14, 'bold'), anchor='w').pack(fill='x')
        tk.Label(self.startFrame, anchor='w', justify='left', wraplength=580,
                 text='Choose a folder. Photos are compared by what they show using a neural '
                      'network, so resized, padded, cropped, re-encoded and edited copies are '
                      'found. You pick which copies to move.'
                 ).pack(fill='x', pady=(5, 10))
        destinations = tk.Frame(self.startFrame)
        destinations.pack(fill='x')
        tk.Radiobutton(destinations, text=TRASH_NAME, variable=self.destinationVar,
                       value='trash').pack(side='left')
        tk.Radiobutton(destinations, text='Deleted folder (preserve folders)',
                       variable=self.destinationVar, value='folder').pack(side='left')

        modelBox = tk.LabelFrame(self.startFrame, text='Model', padx=8, pady=4)
        modelBox.pack(fill='x')
        for key, spec in MODELS.items():
            suffix = ' - recommended' if key == DEFAULT_MODEL else ''
            tk.Radiobutton(modelBox, text=spec['label'] + suffix, value=key, anchor='w',
                           variable=self.modelVar, command=self._onModelChange).pack(fill='x')

        tk.Scale(self.startFrame, variable=self.thresholdVar, from_=0.60, to=0.99,
                 resolution=0.01, orient='horizontal', length=580,
                 label='Similarity threshold: lower = more matches (more false positives), '
                       'higher = only very close copies').pack(fill='x', pady=(8, 0))
        tk.Checkbutton(self.startFrame, variable=self.rotationsVar, anchor='w',
                       text='Also match rotated copies (90°/180°/270°) - about 4x slower'
                       ).pack(fill='x', pady=(4, 6))

        tk.Label(self.startFrame, text=self._acceleratorText(), anchor='w').pack(fill='x')
        if IS_WINDOWS and self.gpuNames and 'DmlExecutionProvider' not in availableProviders():
            tipBox = tk.Frame(self.startFrame)
            tipBox.pack(fill='x', pady=(2, 0))
            tk.Label(tipBox, anchor='w', justify='left', wraplength=440, fg='#a05a00',
                     text=f'GPU detected: {", ".join(self.gpuNames)}. Install onnxruntime-directml '
                          'to scan several times faster.').pack(side='left', fill='x', expand=True)
            tk.Button(tipBox, text='Copy install command',
                      command=self._copyDirectmlCommand).pack(side='right')

        if statusText:
            tk.Label(self.startFrame, text=statusText, fg='grey', anchor='w').pack(fill='x', pady=(6, 0))
        buttons = tk.Frame(self.startFrame)
        buttons.pack(side='bottom', fill='x')
        chooseButton = tk.Button(buttons, text='Choose folder…', width=18,
                                 command=self.chooseFolder, default='active')
        chooseButton.pack(side='left')
        tk.Button(buttons, text='Quit', width=10, command=self.root.destroy).pack(side='right')
        self.root.bind('<Return>', lambda _e: self.chooseFolder())
        chooseButton.focus_set()

    def chooseFolder(self):
        srcDir = filedialog.askdirectory(title='Select folder to search for similar photos',
                                         mustexist=True, parent=self.root)
        self._bringToFront()
        if not srcDir:
            return
        srcDir = os.path.abspath(srcDir)
        deletedFolder = None
        if self.destinationVar.get() == 'folder':
            deletedFolder = chooseDeletedFolder(self.root, srcDir)
            self._bringToFront()
            if not deletedFolder:
                return
        self.deletedFolder = deletedFolder
        self.startScan(srcDir)

    def startScan(self, srcDir, manualModelPath=None):
        self.lastFolder = srcDir
        self.root.unbind('<Return>')
        self.startFrame.destroy()
        self.root.geometry('640x200')
        ScanWindow(self.root, srcDir, self.onScanDone, self.modelVar.get(),
                   round(self.thresholdVar.get(), 2), self.rotationsVar.get(), manualModelPath)

    def _askForModelFile(self, error):
        spec = MODELS[error.modelKey]
        retry = messagebox.askyesno(
            APP_TITLE,
            f'The {spec["label"]} model could not be downloaded.\n\n{error}\n\n'
            f'Download it in a browser from:\n{spec["url"]}\n\n'
            f'(or copy {spec["file"]} into a "models" folder next to this script).\n\n'
            'Do you already have the file and want to select it now?',
            parent=self.root)
        if not retry:
            return None
        path = filedialog.askopenfilename(title=f'Select {spec["file"]}', parent=self.root,
                                          filetypes=[('ONNX model', '*.onnx'), ('All files', '*')])
        self._bringToFront()
        if not path:
            return None
        if _sha256(path) != spec['sha256'] and not messagebox.askyesno(
                APP_TITLE, 'This file is not the exact expected model version. Use it anyway?',
                parent=self.root):
            return None
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        cached = os.path.join(MODEL_CACHE_DIR, spec['file'])
        if os.path.abspath(path) != os.path.abspath(cached):
            shutil.copyfile(path, cached)
        return cached

    def onScanDone(self, status, payload):
        if status == 'model-missing':
            self.showStart()
            modelPath = self._askForModelFile(payload)
            if modelPath:
                self.startScan(self.lastFolder, modelPath)
            else:
                self.startFrame.destroy()
                self.showStart('Model not available - scan cancelled.')
            return
        if status == 'error':
            messagebox.showerror(APP_TITLE, f'Scan failed:\n{payload}', parent=self.root)
            self.showStart('Last scan failed.')
            return
        groups, unreadable, similarity, providerName = payload
        if unreadable:
            shown = '\n'.join(f'{os.path.basename(p)}: {reason}'
                              for p, reason in list(unreadable.items())[:10])
            more = f'\n… and {len(unreadable) - 10} more' if len(unreadable) > 10 else ''
            hint = ('\n\nHEIC/HEIF photos need: pip install pillow-heif'
                    if any(p.lower().endswith(('.heic', '.heif')) for p in unreadable) else '')
            messagebox.showwarning(APP_TITLE, f'{len(unreadable)} file(s) could not be read '
                                   f'and were not compared:\n\n{shown}{more}{hint}',
                                   parent=self.root)
        if not groups:
            messagebox.showinfo(APP_TITLE, 'No similar photos found.', parent=self.root)
            self.showStart(f'Last scan ({providerName}): no similar photos found.')
            return
        self.root.geometry('1100x800')
        self._bringToFront()
        DuplicateReviewer(self.root, groups, self.onReviewDone, self.lastFolder,
                          self.deletedFolder, similarity)

    def onReviewDone(self, deletedCount):
        messagebox.showinfo(APP_TITLE, f'Review finished.\nDeleted {deletedCount} file(s).',
                            parent=self.root)
        destination = self.deletedFolder or f'the {TRASH_NAME}'
        self.showStart(f'Last review: {deletedCount} file(s) moved to {destination}.')


if __name__ == '__main__':
    DedupeApp().run()
