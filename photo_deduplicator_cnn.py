"""Find similar photos (pHash or CNN), or check a subset folder against a set folder.

Home screen:
    Search by pHash         64-bit perceptual hash. Very fast, no extra packages or models.
                            Finds resized, re-encoded and lightly edited copies.
    Search by CNN           1280-number feature vectors from a pretrained image network
                            (ONNX Runtime). Also finds padded, cropped, filtered copies and
                            near-identical shots of the same scene.
    Find Subset in a Set    Compares every photo of a "subset" folder with a "set" folder using
                            SHA-256 (default, exact) or pHash (visual). Shows a matching list and
                            a non-matching list with keep/delete recommendations for the subset.
                            Nothing is deleted or changed, and the set is only read. The saved
                            report also contains a deletion script (text only, never executed).

For the two search modes, selected photos can be moved to Trash or to a "deleted" folder
(suggested next to the scanned folder, editable), preserving their paths relative to the
scanned folder.

CNN models (downloaded once from the official ONNX Model Zoo on GitHub, or put the
.onnx file in a "models" folder next to this script for offline use):
    EfficientNet-Lite4  52 MB  most accurate (default)
    MobileNetV2         14 MB  about 3x faster, slightly less accurate

CNN acceleration is detected automatically:
    Windows  DirectML GPU (needs onnxruntime-directml; suggested when a GPU is found)
    macOS    Core ML (Apple GPU / Neural Engine, included in onnxruntime)
    else     CPU

Requires: numpy, Pillow. For CNN also: onnxruntime, onnx (onnx is only used once
per model to prepare it). Optional: send2trash, pillow-heif, truststore.

Run:
    Windows (no console window):  pythonw photo_deduplicator_combined.py
    macOS / Linux (e.g. Mint):    python3 photo_deduplicator_combined.py
"""
import hashlib
import importlib.util
import logging
import os
import platform
import queue
import shlex
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
GRID_COLUMNS = 3
MIN_PREVIEW_SIZE = 150
MAX_PREVIEW_SIZE = 1600
# Previews are decoded once per group at this size, then scaled for the slider/window.
PREVIEW_CACHE_SIZE = 1600
# Space each photo cell needs besides the image: border, padding, grid spacing, text lines.
PREVIEW_CELL_EXTRA_WIDTH = 30
PREVIEW_CELL_EXTRA_HEIGHT = 130
DECODE_WORKERS = min(8, max(1, os.cpu_count() or 1))
DECODE_CHUNK = 32
APP_TITLE = 'Photo Deduplicator'
MODEL_INPUT_SIZE = 224

METHOD_PHASH = 'phash'
METHOD_CNN = 'cnn'
METHOD_SUBSET = 'subset'
COMPARE_SHA256 = 'sha256'
SUBSET_DEFAULT_MAX_BITS = 4   # stricter than the search default: a false match hides a unique photo
WINDOW_FRACTION = 0.8         # every screen uses this share of the screen width and height

# pHash: 32x32 greyscale -> DCT -> top-left 8x8 coefficients -> 64 bits.
PHASH_SIZE = 32
PHASH_LOW = 8
PHASH_BITS = PHASH_LOW * PHASH_LOW
PHASH_DEFAULT_MAX_BITS = 8

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

def loadGrayPixels(imagePath, size):
    """Whole photo (no crop) as a size x size greyscale uint8 array, EXIF-rotated,
    transparency on white. Greyscale makes black-and-white or colour-adjusted copies
    match their originals."""
    with Image.open(imagePath) as im:
        # Fast JPEG downscale on load; no-op for other formats.
        im.draft('RGB', (size * 2, size * 2))
        im = ImageOps.exif_transpose(im)
        if im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in im.info:
            rgba = im.convert('RGBA')
            im = Image.new('RGBA', rgba.size, 'white')
            im.alpha_composite(rgba)
        im = im.convert('L').resize((size, size), Image.Resampling.BICUBIC, reducing_gap=3.0)
        return np.asarray(im)


def loadModelPixels(imagePath):
    """Greyscale 224x224 replicated to 3 channels for the CNN."""
    gray = loadGrayPixels(imagePath, MODEL_INPUT_SIZE)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _loadVariants(imagePath, embedder, matchRotations):
    try:
        pixels = embedder.loadPixels(imagePath)
    except Exception as exc:
        logging.warning('Cannot read %s: %s', imagePath, exc)
        return imagePath, None, str(exc)
    turns = range(4) if matchRotations else range(1)
    return imagePath, [np.rot90(pixels, k) for k in turns], None


class PhashEmbedder:
    """64-bit DCT perceptual hash, stored as a unit-length vector of +-1/8 values.

    The cosine of two such vectors is 1 - hammingDistance/32, so the same
    similarity search as the CNN applies and a Hamming limit maps to a threshold.
    """
    provider = 'CPUExecutionProvider'

    def __init__(self):
        n = PHASH_SIZE
        k = np.arange(n)[:, None]
        x = np.arange(n)[None, :]
        self.dct = np.cos(np.pi * (2 * x + 1) * k / (2 * n))

    def loadPixels(self, imagePath):
        return loadGrayPixels(imagePath, PHASH_SIZE)

    def embed(self, images):
        vectors = []
        for image in images:
            coefficients = self.dct @ image.astype(np.float64) @ self.dct.T
            low = coefficients[:PHASH_LOW, :PHASH_LOW].reshape(-1)
            bits = low > np.median(low[1:])  # median without the DC term
            vectors.append(np.where(bits, 1.0, -1.0) / np.sqrt(PHASH_BITS))
        return np.stack(vectors).astype(np.float32)


def phashThreshold(maxBits):
    """Cosine threshold equivalent to 'at most maxBits of 64 bits differ'."""
    return 1.0 - maxBits * 2.0 / PHASH_BITS - 0.01  # margin for float rounding


class Embedder:
    def __init__(self, modelKey, modelPath):
        self.spec = MODELS[modelKey]
        self.session = createSession(modelPath)
        self.inputName = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]

    def loadPixels(self, imagePath):
        return loadModelPixels(imagePath)

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
                    lambda p: _loadVariants(p, embedder, matchRotations), chunk):
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


def findSimilarGroups(srcDir, method, modelKey, threshold, matchRotations,
                      onProgress=None, onStatus=None, manualModelPath=None):
    """Return (groups, {unreadablePath: reason}, describeFn, providerName).

    threshold is a cosine similarity for CNN and the maximum number of differing
    bits (of 64) for pHash. describeFn(pathA, pathB) returns text for the reviewer.
    """
    onStatus = onStatus or (lambda _text: None)
    if method == METHOD_PHASH:
        embedder = PhashEmbedder()
        cosineThreshold = phashThreshold(threshold)
    else:
        modelPath = ensureModel(modelKey, onStatus, manualModelPath)
        onStatus('Loading model...')
        embedder = Embedder(modelKey, modelPath)
        cosineThreshold = threshold
    providerName = PROVIDER_NAMES.get(embedder.provider, embedder.provider)
    onStatus(f'Collecting images... (running on {providerName})')
    imagePaths = collectImages(srcDir)
    if onProgress:
        onProgress(0, len(imagePaths))
    paths, embeddings, unreadable = computeEmbeddings(imagePaths, embedder, matchRotations, onProgress)
    if not paths:
        return [], unreadable, None, providerName
    groups = buildGroups(findDuplicates(paths, embeddings, cosineThreshold))
    index = {p: i for i, p in enumerate(paths)}

    def describe(pathA, pathB):
        i, j = index[pathA], index[pathB]
        cosine = float(similarityMatrixRows(embeddings, i, i + 1)[0, j])
        if method == METHOD_PHASH:
            return f'{max(0, round((1 - cosine) * PHASH_BITS / 2))} of {PHASH_BITS} bits differ'
        return f'similarity {cosine:.0%}'

    return groups, unreadable, describe, providerName


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


# --------------------------------------------------------------- subset in a set

def _hashFile(path):
    try:
        return path, _sha256(path), None
    except Exception as exc:
        logging.warning('Cannot hash %s: %s', path, exc)
        return path, None, str(exc)


def _hashFiles(paths, label, onProgress, offset, total, unreadable):
    """SHA-256 of each file -> {path: hexdigest}; files that cannot be read go to `unreadable`."""
    hashes = {}
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        for start in range(0, len(paths), 16):
            chunk = paths[start:start + 16]
            for path, digest, error in pool.map(_hashFile, chunk):
                if digest is None:
                    unreadable[path] = error
                else:
                    hashes[path] = digest
            if onProgress:
                onProgress(offset + start + len(chunk), total, label)
    return hashes


def bestMatches(queryEmb, targetEmb, chunkRows=128):
    """(targetIndex, cosine) of the most similar target photo for every query photo."""
    queryCount, variants, dims = queryEmb.shape
    targetCount = targetEmb.shape[0]
    bestIndex = np.zeros(queryCount, dtype=np.int64)
    bestScore = np.full(queryCount, -1.0, dtype=np.float32)
    if targetCount == 0:
        return bestIndex, bestScore
    queryBase = queryEmb[:, 0, :]
    targetBase = targetEmb[:, 0, :]
    targetAll = targetEmb.reshape(targetCount * variants, dims)
    for start in range(0, queryCount, chunkRows):
        end = min(start + chunkRows, queryCount)
        forward = (queryBase[start:end] @ targetAll.T).reshape(
            end - start, targetCount, variants).max(axis=2)
        backward = (queryEmb[start:end].reshape(-1, dims) @ targetBase.T).reshape(
            end - start, variants, targetCount).max(axis=1)
        sims = np.maximum(forward, backward)
        best = sims.argmax(axis=1)
        bestIndex[start:end] = best
        bestScore[start:end] = sims[np.arange(end - start), best]
    return bestIndex, bestScore


def recommendForMatch(method, subsetPath, setPath):
    """('Delete' | 'Keep', reason) for a subset photo that has a match in the set."""
    if method == COMPARE_SHA256:
        return 'Delete', 'identical file exists in the set'
    subsetWidth, subsetHeight = imgRes(subsetPath)
    setWidth, setHeight = imgRes(setPath)
    if subsetWidth * subsetHeight > setWidth * setHeight:
        return 'Keep', (f'higher resolution than its match in the set '
                        f'({subsetWidth}x{subsetHeight} vs {setWidth}x{setHeight})')
    return 'Delete', 'its match in the set has the same or a higher resolution'


def findSubsetMatches(subsetDir, setDir, method, maxBits, matchRotations,
                      onProgress=None, onStatus=None):
    """Compare every photo of subsetDir with setDir. Only reads files; changes nothing.

    Returns a result dict: matches (subset photos found in the set, each with a
    recommendation), nonMatches [(path, reason)], counts and unreadable files.
    """
    onStatus = onStatus or (lambda _text: None)
    report = onProgress or (lambda *_args: None)
    onStatus('Collecting images...')
    subsetPaths = sorted(collectImages(subsetDir))
    setPaths = sorted(collectImages(setDir))
    unreadable = {}
    matchMap = {}   # subset path -> (set path, detail)

    if method == COMPARE_SHA256:
        # Identical files have identical sizes, so only files whose size occurs on both
        # sides need hashing.
        subsetSizes = {p: fileSize(p) for p in subsetPaths}
        setSizes = {p: fileSize(p) for p in setPaths}
        sizesInSet, sizesInSubset = set(setSizes.values()), set(subsetSizes.values())
        subsetCandidates = [p for p in subsetPaths if subsetSizes[p] in sizesInSet]
        setCandidates = [p for p in setPaths if setSizes[p] in sizesInSubset]
        total = len(subsetCandidates) + len(setCandidates)
        report(0, total, 'Checking files')
        setHashes = _hashFiles(setCandidates, 'Checking files in the set', report,
                               0, total, unreadable)
        subsetHashes = _hashFiles(subsetCandidates, 'Checking files in the subset', report,
                                  len(setCandidates), total, unreadable)
        setByHash = {}
        for path, digest in setHashes.items():
            setByHash.setdefault(digest, path)
        for path, digest in subsetHashes.items():
            if digest in setByHash:
                matchMap[path] = (setByHash[digest], 'identical (SHA-256)')
    else:
        embedder = PhashEmbedder()
        total = len(subsetPaths) + len(setPaths)
        subsetOk, subsetEmb, subsetBad = computeEmbeddings(
            subsetPaths, embedder, matchRotations,
            lambda done, _count: report(done, total, 'Analysing subset photos'))
        setOk, setEmb, setBad = computeEmbeddings(
            setPaths, embedder, matchRotations,
            lambda done, _count: report(len(subsetPaths) + done, total, 'Analysing set photos'))
        unreadable.update(subsetBad)
        unreadable.update(setBad)
        if subsetOk and setOk:
            bestIndex, bestScore = bestMatches(subsetEmb, setEmb)
            threshold = phashThreshold(maxBits)
            for path, index, score in zip(subsetOk, bestIndex, bestScore):
                if score >= threshold:
                    bits = max(0, round((1 - float(score)) * PHASH_BITS / 2))
                    matchMap[path] = (setOk[index], f'{bits} of {PHASH_BITS} bits differ')

    matches, nonMatches = [], []
    for path in subsetPaths:
        if path in matchMap:
            setPath, detail = matchMap[path]
            action, reason = recommendForMatch(method, path, setPath)
            matches.append({'path': path, 'setPath': setPath, 'detail': detail,
                            'action': action, 'reason': reason})
        elif path in unreadable:
            nonMatches.append((path, f'could not be read ({unreadable[path]}), not checked'))
        else:
            nonMatches.append((path, 'not found in the set'))
    return {'method': method, 'subsetDir': subsetDir, 'setDir': setDir,
            'subsetCount': len(subsetPaths), 'setCount': len(setPaths),
            'matches': matches, 'nonMatches': nonMatches, 'unreadable': unreadable}


def subsetCounts(result):
    """(photos recommended for deletion, photos recommended to preserve)."""
    delete = sum(1 for m in result['matches'] if m['action'] == 'Delete')
    return delete, len(result['matches']) + len(result['nonMatches']) - delete


def compareMethodName(method):
    return 'SHA-256 (identical files)' if method == COMPARE_SHA256 else 'pHash (visually similar photos)'


def buildReportText(result):
    delete, keep = subsetCounts(result)
    lines = ['Subset in a set - recommendations only',
             'Nothing was deleted or changed; the set was only read.', '',
             f'Subset folder: {result["subsetDir"]}',
             f'Set folder:    {result["setDir"]}',
             f'Compared by:   {compareMethodName(result["method"])}',
             f'Subset photos: {result["subsetCount"]}   Set photos: {result["setCount"]}',
             f'Recommendation: delete {delete}, preserve {keep} photo(s) in the subset', '',
             f'MATCHING - found in the set ({len(result["matches"])})']
    for match in result['matches']:
        lines.append(f'  {match["action"].upper()}: {match["path"]}')
        lines.append(f'      matches: {match["setPath"]}  [{match["detail"]}]')
        lines.append(f'      reason: {match["reason"]}')
    lines += ['', f'NOT MATCHING ({len(result["nonMatches"])})']
    for path, reason in result['nonMatches']:
        lines.append(f'  PRESERVE: {path}  ({reason})')
    lines += [''] + buildDeletionScript(result)
    return '\n'.join(lines) + '\n'


def _psQuote(path):
    """PowerShell single-quoted literal. PowerShell treats the typographic single quotes
    as quote characters too, and inside a literal each one is escaped by doubling it."""
    for quote in ("'", '\u2018', '\u2019', '\u201a', '\u201b'):
        path = path.replace(quote, quote * 2)
    return "'" + path + "'"


def buildDeletionScript(result):
    """Report section with commands that delete the subset photos recommended for deletion.

    It is only text inside the report: nothing here is executed by this program, and only
    files of the subset folder are listed (never the set).
    """
    subsetDir = result['subsetDir']
    targets = [m['path'] for m in result['matches']
               if m['action'] == 'Delete' and _isWithin(m['path'], subsetDir)]
    if IS_WINDOWS:
        shellName = 'Windows PowerShell'
        fate = 'Files are sent to the Recycle Bin, so they can be restored.'
    else:
        shellName = 'POSIX shell (bash/zsh, macOS, Linux)'
        fate = 'Files are removed permanently, not sent to the Trash.'
    lines = ['DELETION SCRIPT (part of this report - this program never runs it)',
             f'Shell: {shellName}',
             f'Deletes {len(targets)} subset photo(s) recommended for deletion. {fate}',
             'Review the list, then copy the commands you want into a terminal.',
             'The set folder is not touched by any command below.',
             '-' * 70]
    if not targets:
        lines.append('# Nothing is recommended for deletion.')
    elif IS_WINDOWS:
        lines.append('Add-Type -AssemblyName Microsoft.VisualBasic')
    for path in targets:
        if IS_WINDOWS:
            lines.append('[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile('
                         f"{_psQuote(path)}, 'OnlyErrorDialogs', 'SendToRecycleBin')")
        else:
            lines.append('rm -- ' + shlex.quote(path))
    lines.append('-' * 70)
    return lines


def validateSubsetFolders(subsetDir, setDir):
    """Return (subsetAbs, setAbs, None) when usable, otherwise (None, None, reason)."""
    subsetDir, setDir = subsetDir.strip(), setDir.strip()
    if not subsetDir or not setDir:
        return None, None, 'Choose both the subset folder and the set folder.'
    subsetDir = os.path.abspath(os.path.expanduser(subsetDir))
    setDir = os.path.abspath(os.path.expanduser(setDir))
    for name, folder in (('subset', subsetDir), ('set', setDir)):
        if not os.path.isdir(folder):
            return None, None, f'The {name} folder does not exist:\n{folder}'
    if _isWithin(subsetDir, setDir) or _isWithin(setDir, subsetDir):
        return None, None, ('The subset and set folders must be separate '
                            '(neither may be inside the other).')
    return subsetDir, setDir, None


# -------------------------------------------------------------------------- GUI

def screenGeometry(root, fraction=WINDOW_FRACTION):
    """Window size/position for `fraction` of the screen, centred.

    Uses real pixels (the process is DPI-aware on Windows), so the window covers the same
    share of the screen at any resolution or scaling.
    """
    screenWidth, screenHeight = root.winfo_screenwidth(), root.winfo_screenheight()
    width, height = int(screenWidth * fraction), int(screenHeight * fraction)
    return f'{width}x{height}+{(screenWidth - width) // 2}+{(screenHeight - height) // 2}'


def autoWrap(label, margin=10):
    """Make a label that fills its parent's width re-wrap its text when the window resizes."""
    label.bind('<Configure>', lambda event: label.configure(wraplength=max(200, event.width - margin)))
    return label


class ScanWindow:
    """Shows progress while `work(onProgress, onStatus)` runs in a background thread.

    onProgress(done, total, label=None) and onStatus(text) may be called from the worker.
    onDone(status, payload) is called with 'done', 'model-missing' or 'error'.
    """

    def __init__(self, root, onDone, work, heading, detail):
        self.root = root
        self.onDone = onDone
        self.messages = queue.Queue()

        self.frame = tk.Frame(root, padx=20, pady=20)
        self.frame.pack(fill='both', expand=True)
        tk.Label(self.frame, text=heading, font=(UI_FONT_FAMILY, 11, 'bold'),
                 anchor='w').pack(fill='x')
        autoWrap(tk.Label(self.frame, text=detail, anchor='w', justify='left')
                 ).pack(fill='x', pady=(0, 10))
        self.statusLabel = tk.Label(self.frame, text='Preparing...', anchor='w')
        self.statusLabel.pack(fill='x')
        self.progress = ttk.Progressbar(self.frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=10)
        self.progress.start(10)
        self.providerName = ''
        self.stageText = 'Analysing images'

        threading.Thread(target=self._worker, args=(work,), daemon=True).start()
        self.root.after(100, self._poll)

    def _worker(self, work):
        try:
            result = work(
                lambda done, total, label=None: self.messages.put(('progress', done, total, label)),
                lambda text: self.messages.put(('status', text)))
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
                    _, done, total, label = msg
                    if label:
                        self.stageText = label
                    if self.progress['mode'] != 'determinate':
                        self.progress.stop()
                        self.progress.config(mode='determinate', maximum=max(total, 1))
                    self.progress['value'] = done
                    running = f'  (running on {self.providerName})' if self.providerName else ''
                    self.statusLabel.config(text=f'{self.stageText}: {done} of {total}{running}')
                else:
                    self.frame.destroy()
                    self.onDone(msg[0], msg[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


class SubsetReport:
    """Matching and non-matching lists with keep/delete recommendations. Changes nothing."""

    def __init__(self, root, result, onFinish):
        self.root = root
        self.result = result
        self.onFinish = onFinish
        matches, nonMatches = result['matches'], result['nonMatches']
        delete, keep = subsetCounts(result)

        self.frame = tk.Frame(root)
        self.frame.pack(fill='both', expand=True)
        tk.Label(self.frame, text='Subset in a set - recommendations',
                 font=(UI_FONT_FAMILY, 12, 'bold'), anchor='w').pack(fill='x', padx=10, pady=(10, 0))
        summary = (f'Compared by {compareMethodName(result["method"])}. '
                   f'{len(matches)} of {result["subsetCount"]} subset photos were found in the set; '
                   f'{len(nonMatches)} were not.\n'
                   f'Recommendation: delete {delete} and preserve {keep} photo(s) in the subset. '
                   'Nothing has been deleted or changed, and the set was only read.')
        autoWrap(tk.Label(self.frame, text=summary, anchor='w', justify='left')
                 ).pack(fill='x', padx=10, pady=(0, 6))

        panes = tk.PanedWindow(self.frame, orient='horizontal', sashrelief='raised', sashwidth=6)
        panes.pack(fill='both', expand=True, padx=10)
        subsetDir, setDir = result['subsetDir'], result['setDir']
        self._addList(
            panes, f'Matching - already in the set ({len(matches)})',
            [('Subset photo', 260), ('Matching photo in the set', 260),
             ('Match', 120), ('Recommendation', 300)],
            [((os.path.relpath(m['path'], subsetDir), os.path.relpath(m['setPath'], setDir),
               m['detail'], f'{m["action"]} - {m["reason"]}'), m['action'].lower())
             for m in matches])
        self._addList(
            panes, f'Not matching - not in the set ({len(nonMatches)})',
            [('Subset photo', 300), ('Recommendation', 260)],
            [((os.path.relpath(path, subsetDir), f'Preserve - {reason}'), 'keep')
             for path, reason in nonMatches])

        buttons = tk.Frame(self.frame)
        buttons.pack(fill='x', padx=10, pady=10)
        tk.Button(buttons, text='View report…', width=16, command=self.viewReport).pack(side='left')
        tk.Button(buttons, text='Save report…', width=16,
                  command=self.saveReport).pack(side='left', padx=8)
        tk.Button(buttons, text='Done', width=12, command=self.finish).pack(side='right')

    def _addList(self, panes, title, columns, rows):
        box = tk.LabelFrame(panes, text=title, padx=4, pady=4)
        panes.add(box, stretch='always', minsize=300)
        names = [f'c{i}' for i in range(len(columns))]
        tree = ttk.Treeview(box, columns=names, show='headings', selectmode='extended')
        for name, (heading, width) in zip(names, columns):
            tree.heading(name, text=heading)
            tree.column(name, width=width, minwidth=80, stretch=True)
        vbar = ttk.Scrollbar(box, orient='vertical', command=tree.yview)
        hbar = ttk.Scrollbar(box, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        box.grid_rowconfigure(0, weight=1)
        box.grid_columnconfigure(0, weight=1)
        tree.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='ew')
        tree.tag_configure('delete', foreground='#b03a00')
        tree.tag_configure('keep', foreground='#1b6e1b')
        for values, tag in rows:
            tree.insert('', 'end', values=values, tags=(tag,))

    def viewReport(self):
        """Show the full report (including the deletion script) before saving."""
        viewer = tk.Toplevel(self.root)
        viewer.title('Report preview')
        viewer.transient(self.root)
        viewer.geometry(screenGeometry(self.root))
        tk.Label(viewer, anchor='w', justify='left',
                 text='This is exactly what "Save report" writes. Nothing is executed.'
                 ).pack(fill='x', padx=10, pady=(10, 4))
        body = tk.Frame(viewer)
        body.pack(fill='both', expand=True, padx=10)
        text = tk.Text(body, wrap='none', font=('Courier', 10))
        vbar = tk.Scrollbar(body, orient='vertical', command=text.yview)
        hbar = tk.Scrollbar(body, orient='horizontal', command=text.xview)
        text.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        text.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='ew')
        text.insert('1.0', buildReportText(self.result))
        text.configure(state='disabled')  # read-only, still selectable and copyable
        buttons = tk.Frame(viewer)
        buttons.pack(fill='x', padx=10, pady=10)
        tk.Button(buttons, text='Save report…', width=16,
                  command=lambda: self.saveReport(viewer)).pack(side='left')
        tk.Button(buttons, text='Close', width=12, command=viewer.destroy).pack(side='right')
        viewer.bind('<Escape>', lambda _e: viewer.destroy())
        viewer.focus_set()

    def saveReport(self, parent=None):
        parent = parent or self.root
        path = filedialog.asksaveasfilename(
            parent=parent, title='Save report', defaultextension='.txt',
            initialfile='subset_report.txt',
            filetypes=[('Text file', '*.txt'), ('All files', '*')])
        if not path:
            return
        if _isWithin(os.path.dirname(os.path.abspath(path)), self.result['setDir']):
            messagebox.showerror('Save report', 'Choose a location outside the set folder - '
                                 'the set is never modified.', parent=parent)
            return
        try:
            with open(path, 'w', encoding='utf-8') as out:
                out.write(buildReportText(self.result))
        except OSError as exc:
            messagebox.showerror('Save report', f'Could not save the report:\n{exc}', parent=parent)
            return
        messagebox.showinfo('Save report', f'Report saved to:\n{path}', parent=parent)

    def finish(self):
        self.frame.destroy()
        self.onFinish()


def fitPreviewLayout(availableWidth, availableHeight, photoCount):
    """Columns and preview size so a group's photos fill the visible area (up to 2 rows)."""
    columns = max(1, min(photoCount, GRID_COLUMNS))
    rows = -(-photoCount // columns)
    byWidth = availableWidth // columns - PREVIEW_CELL_EXTRA_WIDTH
    byHeight = availableHeight // min(rows, 2) - PREVIEW_CELL_EXTRA_HEIGHT
    return columns, max(MIN_PREVIEW_SIZE, min(MAX_PREVIEW_SIZE, byWidth, byHeight))


def loadPreview(imagePath):
    """Photo as displayed (EXIF-rotated) at up to PREVIEW_CACHE_SIZE pixels."""
    with Image.open(imagePath) as im:
        # Fast JPEG downscale on load; no-op for other formats.
        im.draft('RGB', (PREVIEW_CACHE_SIZE, PREVIEW_CACHE_SIZE))
        im = ImageOps.exif_transpose(im).convert('RGB')
    im.thumbnail((PREVIEW_CACHE_SIZE, PREVIEW_CACHE_SIZE), Image.Resampling.LANCZOS)
    return im


class DuplicateReviewer:
    def __init__(self, root, groups, onFinish, srcDir, deletedFolder, describe=None):
        self.root = root
        self.groups = groups
        self.onFinish = onFinish
        self.srcDir = srcDir
        self.deletedFolder = deletedFolder
        self.describe = describe
        self.groupIndex = 0
        self.deletedCount = 0
        self.thumbRefs = []
        self.checkVars = []
        self.cells = []
        self.previewCache = {}
        self.currentLayout = None
        self.currentSize = None
        self.resizeJob = None
        self.fitVar = tk.BooleanVar(value=True)
        self.sizeVar = tk.IntVar(value=400)

        self.frame = tk.Frame(root)
        self.frame.pack(fill='both', expand=True)

        self.headerLabel = tk.Label(self.frame, font=(UI_FONT_FAMILY, 12, 'bold'), anchor='w')
        self.headerLabel.pack(fill='x', padx=10, pady=(10, 0))
        tk.Label(self.frame, anchor='w',
                 text='Tick the photos to delete (or click a photo). If nothing is ticked, '
                      'the group is skipped. Similarity is compared with photo 1. '
                      'Use "View larger" or the Photo size slider for details.'
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
        self.canvas.bind('<Configure>', lambda _e: self._scheduleRender())
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
        sizeBox = tk.Frame(buttons)
        sizeBox.pack(side='right', padx=12)
        tk.Checkbutton(sizeBox, text='Fit to window', variable=self.fitVar,
                       command=lambda: self._scheduleRender(0)).pack(side='left', anchor='s')
        tk.Scale(sizeBox, variable=self.sizeVar, from_=MIN_PREVIEW_SIZE, to=MAX_PREVIEW_SIZE,
                 resolution=50, orient='horizontal', length=240, label='Photo size',
                 command=self._onSizeSlider).pack(side='left', padx=(8, 0))
        self.root.bind('<Return>', lambda _e: self.deleteAndNext())

        self.showGroup()

    def _onMouseWheel(self, event):
        # Windows reports multiples of 120 per notch; macOS reports small raw deltas.
        step = -int(event.delta / 120) if IS_WINDOWS else -event.delta
        if step:
            self.canvas.yview_scroll(step, 'units')

    def _onSizeSlider(self, value):
        if int(float(value)) == self.currentSize:
            return
        self.fitVar.set(False)
        self._scheduleRender(100)

    def _scheduleRender(self, delay=150):
        if self.resizeJob:
            self.root.after_cancel(self.resizeJob)
        self.resizeJob = self.root.after(delay, self._renderPreviews)

    def _renderPreviews(self):
        self.resizeJob = None
        if not self.cells or not self.canvas.winfo_exists():
            return
        self.canvas.update_idletasks()
        width = max(self.canvas.winfo_width(), 300)
        height = max(self.canvas.winfo_height(), 300)
        columns, size = fitPreviewLayout(width, height, len(self.cells))
        if not self.fitVar.get():
            size = self.sizeVar.get()
            columns = max(1, min(len(self.cells), width // (size + PREVIEW_CELL_EXTRA_WIDTH)))
        if (columns, size) == self.currentLayout:
            return
        self.currentLayout = (columns, size)
        self.currentSize = size
        if self.fitVar.get():
            self.sizeVar.set(size)
        self.thumbRefs.clear()
        for idx, (cell, imageLabel, textWidgets, path) in enumerate(self.cells):
            cell.grid(row=idx // columns, column=idx % columns, padx=5, pady=5, sticky='n')
            source = self.previewCache.get(path)
            if source is not None:
                preview = source.copy()
                preview.thumbnail((size, size), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(preview)
                self.thumbRefs.append(photo)
                imageLabel.configure(image=photo)
            for widget in textWidgets:
                widget.configure(wraplength=max(size, 200))

    def _openViewer(self, path):
        """Show one photo as large as the screen allows; click or Esc closes it."""
        screenWidth, screenHeight = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((int(screenWidth * 0.95), int(screenHeight * 0.85)),
                         Image.Resampling.LANCZOS)
        except Exception as exc:
            messagebox.showerror('Cannot open image', f'{path}\n{exc}', parent=self.root)
            return
        viewer = tk.Toplevel(self.root)
        viewer.title(f'{os.path.basename(path)}  ({imgRes(path)[0]}x{imgRes(path)[1]})')
        photo = ImageTk.PhotoImage(im)
        imageLabel = tk.Label(viewer, image=photo, cursor='hand2')
        imageLabel.image = photo
        imageLabel.pack()
        tk.Label(viewer, text=f'{path}   —   click the photo or press Esc to close',
                 fg='grey').pack(pady=4)
        imageLabel.bind('<Button-1>', lambda _e: viewer.destroy())
        viewer.bind('<Escape>', lambda _e: viewer.destroy())
        viewer.lift()
        viewer.focus_force()

    def showGroup(self):
        if self.resizeJob:
            self.root.after_cancel(self.resizeJob)
            self.resizeJob = None
        for child in self.gridFrame.winfo_children():
            child.destroy()
        self.thumbRefs.clear()
        self.checkVars.clear()
        self.cells.clear()
        self.previewCache.clear()
        self.currentLayout = None
        self.canvas.yview_moveto(0)

        if self.groupIndex >= len(self.groups):
            self.finish()
            return

        group = self.groups[self.groupIndex]
        self.headerLabel.config(
            text=f'Group {self.groupIndex + 1} of {len(self.groups)} — {len(group)} similar photos')

        for idx, path in enumerate(group):
            cell = tk.Frame(self.gridFrame, bd=1, relief='solid', padx=5, pady=5)
            try:
                self.previewCache[path] = loadPreview(path)
                imageLabel = tk.Label(cell, cursor='hand2')
            except Exception as exc:
                imageLabel = tk.Label(cell, text=f'Cannot open image\n{exc}',
                                      width=40, height=15)
            imageLabel.pack()

            width, height = imgRes(path)
            if idx == 0:
                note = '  (largest)'
            elif self.describe:
                note = f'  {self.describe(group[0], path)}'
            else:
                note = ''
            checkVar = tk.BooleanVar(value=False)
            self.checkVars.append((checkVar, path))
            imageLabel.bind('<Button-1>', lambda _e, v=checkVar: v.set(not v.get()))
            checkButton = tk.Checkbutton(
                cell, variable=checkVar, justify='left',
                text=f'{idx + 1}. {os.path.basename(path)}\n'
                     f'{width}x{height}, {fileSize(path) / 1024:.0f} KB{note}')
            checkButton.pack(anchor='w')
            folderLabel = tk.Label(cell, text=os.path.dirname(path), fg='grey', justify='left')
            folderLabel.pack(anchor='w')
            tk.Button(cell, text='View larger',
                      command=lambda p=path: self._openViewer(p)).pack(anchor='w', pady=(2, 0))
            self.cells.append((cell, imageLabel, (checkButton, folderLabel), path))
        self._renderPreviews()

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
        if self.resizeJob:
            self.root.after_cancel(self.resizeJob)
            self.resizeJob = None
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
        self.root.title(APP_TITLE)
        self.startFrame = None
        self.method = None            # None = home screen, else METHOD_PHASH / METHOD_CNN / METHOD_SUBSET
        self.lastFolder = None
        self.deletedFolder = None
        self.destinationVar = tk.StringVar(value='trash')
        self.modelVar = tk.StringVar(value=DEFAULT_MODEL)
        self.thresholdVar = tk.DoubleVar(value=MODELS[DEFAULT_MODEL]['defaultThreshold'])
        self.phashBitsVar = tk.IntVar(value=PHASH_DEFAULT_MAX_BITS)
        self.rotationsVar = tk.BooleanVar(value=True)
        self.subsetDirVar = tk.StringVar()
        self.setDirVar = tk.StringVar()
        self.compareVar = tk.StringVar(value=COMPARE_SHA256)
        self.subsetBitsVar = tk.IntVar(value=SUBSET_DEFAULT_MAX_BITS)
        self.subsetRotationsVar = tk.BooleanVar(value=True)
        self.subsetOptionWidgets = []
        self.gpuNames = None
        self.gpuDetectStarted = False

    def _bringToFront(self):
        # Only lift/focus: toggling '-topmost' off on macOS sends the window behind other apps.
        self.root.lift()
        self.root.focus_force()

    def run(self):
        self.root.geometry(screenGeometry(self.root))
        self.showHome()
        self.root.after(200, self._bringToFront)
        self.root.mainloop()

    # ---------------------------------------------------------- shared helpers

    def _discardStartFrame(self):
        self.root.unbind('<Return>')
        if self.startFrame is not None and self.startFrame.winfo_exists():
            self.startFrame.destroy()
        self.startFrame = None

    def _newStartFrame(self):
        self._discardStartFrame()
        self.startFrame = tk.Frame(self.root, padx=20, pady=20)
        self.startFrame.pack(fill='both', expand=True)
        return self.startFrame

    def _addHeading(self, parent, title, description):
        tk.Label(parent, text=title, font=(UI_FONT_FAMILY, 14, 'bold'), anchor='w').pack(fill='x')
        autoWrap(tk.Label(parent, anchor='w', justify='left', text=description)
                 ).pack(fill='x', pady=(5, 10))

    def _addDestinationOptions(self, parent):
        destinations = tk.Frame(parent)
        destinations.pack(fill='x')
        tk.Radiobutton(destinations, text=TRASH_NAME, variable=self.destinationVar,
                       value='trash').pack(side='left')
        tk.Radiobutton(destinations, text='Deleted folder (preserve folders)',
                       variable=self.destinationVar, value='folder').pack(side='left')

    def _addBottomButtons(self, parent, statusText, primaryText='Choose folder…',
                          primaryCommand=None):
        primaryCommand = primaryCommand or self.chooseFolder
        if statusText:
            tk.Label(parent, text=statusText, fg='grey', anchor='w').pack(fill='x', pady=(6, 0))
        buttons = tk.Frame(parent)
        buttons.pack(side='bottom', fill='x')
        primaryButton = tk.Button(buttons, text=primaryText, width=18,
                                  command=primaryCommand, default='active')
        primaryButton.pack(side='left')
        tk.Button(buttons, text='Back', width=10, command=self.showHome).pack(side='left', padx=8)
        tk.Button(buttons, text='Quit', width=10, command=self.root.destroy).pack(side='right')
        self.root.bind('<Return>', lambda _e: primaryCommand())
        primaryButton.focus_set()

    def showCurrentStart(self, statusText=''):
        if self.method == METHOD_PHASH:
            self.showPhashStart(statusText)
        elif self.method == METHOD_CNN:
            self.showCnnStart(statusText)
        elif self.method == METHOD_SUBSET:
            self.showSubsetStart(statusText)
        else:
            self.showHome(statusText)

    def _warnUnreadable(self, unreadable):
        if not unreadable:
            return
        shown = '\n'.join(f'{os.path.basename(p)}: {reason}'
                          for p, reason in list(unreadable.items())[:10])
        more = f'\n… and {len(unreadable) - 10} more' if len(unreadable) > 10 else ''
        hint = ('\n\nHEIC/HEIF photos need: pip install pillow-heif'
                if any(p.lower().endswith(('.heic', '.heif')) for p in unreadable) else '')
        messagebox.showwarning(APP_TITLE, f'{len(unreadable)} file(s) could not be read '
                               f'and were not compared:\n\n{shown}{more}{hint}',
                               parent=self.root)

    # -------------------------------------------------------------- home screen

    def showHome(self, statusText=''):
        self.method = None
        frame = self._newStartFrame()
        self._addHeading(frame, 'Photo Deduplicator',
                         'Choose what you want to do. Searching for similar photos lets you pick '
                         'which copies to move; the subset check only gives recommendations.')
        choices = (
            ('Search by pHash', self.showPhashStart,
             'Perceptual hash. Very fast, nothing extra to install. Finds resized, '
             're-encoded and lightly edited copies.'),
            ('Search by Convolutional Neural Network (CNN)', self.showCnnStart,
             'Slower and needs onnxruntime and a model download, but also finds '
             'padded, cropped, filtered copies and near-identical shots.'),
            ('Find Subset in a Set', self.showSubsetStart,
             'Checks which photos of one folder (the subset) already exist in another '
             'folder (the set), by SHA-256 or pHash, and recommends what to preserve and what '
             'to delete in the subset. Nothing is deleted; the set is never touched.'))
        for text, command, description in choices:
            tk.Button(frame, text=text, font=(UI_FONT_FAMILY, 12, 'bold'),
                      command=command).pack(fill='x', ipady=8)
            autoWrap(tk.Label(frame, anchor='w', justify='left', fg='grey', text=description)
                     ).pack(fill='x', pady=(2, 14))
        if statusText:
            tk.Label(frame, text=statusText, fg='grey', anchor='w').pack(fill='x', pady=(0, 10))
        tk.Button(frame, text='Quit', width=10, command=self.root.destroy).pack(side='bottom', anchor='e')

    # ------------------------------------------------------------ pHash screen

    def showPhashStart(self, statusText=''):
        self.method = METHOD_PHASH
        frame = self._newStartFrame()
        self._addHeading(frame, 'Find similar photos (pHash)',
                         'Each photo is reduced to a 64-bit fingerprint of its overall '
                         'structure; photos whose fingerprints differ in only a few bits are '
                         'grouped. You pick which copies to move.')
        self._addDestinationOptions(frame)
        tk.Scale(frame, variable=self.phashBitsVar, from_=0, to=20, resolution=1,
                 orient='horizontal',
                 label='Max differing bits (of 64): lower = only very close copies, '
                       'higher = more matches (more false positives)').pack(fill='x', pady=(8, 0))
        tk.Checkbutton(frame, variable=self.rotationsVar, anchor='w',
                       text='Also match rotated copies (90°/180°/270°)'
                       ).pack(fill='x', pady=(4, 6))
        tk.Label(frame, text='Runs on the CPU', anchor='w').pack(fill='x')
        self._addBottomButtons(frame, statusText)

    # -------------------------------------------------------------- CNN screen

    def _detectGpu(self):
        self.gpuNames = detectWindowsGpus()
        if self.gpuNames:
            self.root.after(0, self._refreshCnnIfShown)

    def _refreshCnnIfShown(self):
        if (self.method == METHOD_CNN and self.startFrame is not None
                and self.startFrame.winfo_exists()):
            self.showCnnStart()

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

    def showCnnStart(self, statusText=''):
        if importlib.util.find_spec('onnxruntime') is None:
            messagebox.showerror(APP_TITLE, 'onnxruntime is not installed, so CNN search is '
                                 'unavailable.\n\nInstall it with:\npip install -r requirements-cnn.txt',
                                 parent=self.root)
            return
        if IS_WINDOWS and not self.gpuDetectStarted and 'DmlExecutionProvider' not in availableProviders():
            self.gpuDetectStarted = True
            threading.Thread(target=self._detectGpu, daemon=True).start()

        # The folder picker is only opened from a button click in this visible window;
        # on macOS a dialog with no visible parent window loses focus and vanishes when clicked.
        self.method = METHOD_CNN
        frame = self._newStartFrame()
        self._addHeading(frame, 'Find similar photos (CNN)',
                         'Photos are compared by what they show using a neural network, so '
                         'resized, padded, cropped, re-encoded and edited copies are found. '
                         'You pick which copies to move.')
        self._addDestinationOptions(frame)

        modelBox = tk.LabelFrame(frame, text='Model', padx=8, pady=4)
        modelBox.pack(fill='x')
        for key, spec in MODELS.items():
            suffix = ' - recommended' if key == DEFAULT_MODEL else ''
            tk.Radiobutton(modelBox, text=spec['label'] + suffix, value=key, anchor='w',
                           variable=self.modelVar, command=self._onModelChange).pack(fill='x')

        tk.Scale(frame, variable=self.thresholdVar, from_=0.60, to=0.99,
                 resolution=0.01, orient='horizontal',
                 label='Similarity threshold: lower = more matches (more false positives), '
                       'higher = only very close copies').pack(fill='x', pady=(8, 0))
        tk.Checkbutton(frame, variable=self.rotationsVar, anchor='w',
                       text='Also match rotated copies (90°/180°/270°) - about 4x slower'
                       ).pack(fill='x', pady=(4, 6))

        tk.Label(frame, text=self._acceleratorText(), anchor='w').pack(fill='x')
        if IS_WINDOWS and self.gpuNames and 'DmlExecutionProvider' not in availableProviders():
            tipBox = tk.Frame(frame)
            tipBox.pack(fill='x', pady=(2, 0))
            tk.Label(tipBox, anchor='w', justify='left', wraplength=440, fg='#a05a00',
                     text=f'GPU detected: {", ".join(self.gpuNames)}. Install onnxruntime-directml '
                          'to scan several times faster.').pack(side='left', fill='x', expand=True)
            tk.Button(tipBox, text='Copy install command',
                      command=self._copyDirectmlCommand).pack(side='right')
        self._addBottomButtons(frame, statusText)

    # ------------------------------------------------------- subset-in-a-set screen

    def _browseInto(self, variable, title):
        options = {}
        current = variable.get().strip()
        if os.path.isdir(current):
            options['initialdir'] = current
        picked = filedialog.askdirectory(title=title, mustexist=True, parent=self.root, **options)
        self._bringToFront()
        if picked:
            variable.set(os.path.normpath(picked))

    def _addFolderRow(self, parent, labelText, variable, dialogTitle):
        tk.Label(parent, text=labelText, anchor='w').pack(fill='x', pady=(8, 0))
        row = tk.Frame(parent)
        row.pack(fill='x')
        tk.Entry(row, textvariable=variable).pack(side='left', fill='x', expand=True)
        tk.Button(row, text='Browse…',
                  command=lambda: self._browseInto(variable, dialogTitle)).pack(side='left', padx=(6, 0))

    def _updateSubsetOptions(self):
        state = 'normal' if self.compareVar.get() == METHOD_PHASH else 'disabled'
        for widget in self.subsetOptionWidgets:
            widget.configure(state=state)

    def showSubsetStart(self, statusText=''):
        self.method = METHOD_SUBSET
        frame = self._newStartFrame()
        self._addHeading(frame, 'Find Subset in a Set',
                         'Checks which photos of the subset folder already exist in the set '
                         'folder. Nothing is moved or deleted and the set is never modified: you '
                         'get a matching list, a non-matching list and a recommendation of what '
                         'to preserve and what to delete in the subset.')
        self._addFolderRow(frame, 'Subset folder (the photos to check):', self.subsetDirVar,
                           'Select the subset folder')
        self._addFolderRow(frame, 'Set folder (the reference, never touched):', self.setDirVar,
                           'Select the set folder')

        modeBox = tk.LabelFrame(frame, text='Compare by', padx=8, pady=4)
        modeBox.pack(fill='x', pady=(12, 0))
        tk.Radiobutton(modeBox, value=COMPARE_SHA256, variable=self.compareVar, anchor='w',
                       text='SHA-256 - exact: files must be byte-for-byte identical (recommended)',
                       command=self._updateSubsetOptions).pack(fill='x')
        tk.Radiobutton(modeBox, value=METHOD_PHASH, variable=self.compareVar, anchor='w',
                       text='pHash - visual: also resized, re-encoded or lightly edited copies',
                       command=self._updateSubsetOptions).pack(fill='x')
        bitsScale = tk.Scale(modeBox, variable=self.subsetBitsVar, from_=0, to=20, resolution=1,
                             orient='horizontal',
                             label='pHash: max differing bits (of 64): lower = stricter')
        bitsScale.pack(fill='x', pady=(6, 0))
        rotationsCheck = tk.Checkbutton(modeBox, variable=self.subsetRotationsVar, anchor='w',
                                        text='pHash: also match rotated copies (90°/180°/270°)')
        rotationsCheck.pack(fill='x')
        self.subsetOptionWidgets = [bitsScale, rotationsCheck]
        self._updateSubsetOptions()
        self._addBottomButtons(frame, statusText, primaryText='Start comparison',
                               primaryCommand=self.startSubsetScan)

    def startSubsetScan(self):
        subsetDir, setDir, error = validateSubsetFolders(self.subsetDirVar.get(),
                                                         self.setDirVar.get())
        if error:
            messagebox.showerror(APP_TITLE, error, parent=self.root)
            return
        method = self.compareVar.get()
        maxBits = int(self.subsetBitsVar.get())
        rotations = self.subsetRotationsVar.get()
        self._discardStartFrame()

        def work(onProgress, onStatus):
            return findSubsetMatches(subsetDir, setDir, method, maxBits, rotations,
                                     onProgress=onProgress, onStatus=onStatus)

        ScanWindow(self.root, self.onSubsetDone, work, 'Comparing the subset with the set:',
                   f'Subset: {subsetDir}\nSet: {setDir}')

    def onSubsetDone(self, status, payload):
        if status != 'done':
            messagebox.showerror(APP_TITLE, f'Comparison failed:\n{payload}', parent=self.root)
            self.showSubsetStart('Last comparison failed.')
            return
        self._warnUnreadable(payload['unreadable'])
        self._bringToFront()
        SubsetReport(self.root, payload, lambda: self.showSubsetStart(
            f'Last comparison: {len(payload["matches"])} of {payload["subsetCount"]} '
            'subset photos found in the set.'))

    # ------------------------------------------------------------ scan / review

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
        self._discardStartFrame()
        method = self.method
        modelKey = self.modelVar.get()
        rotations = self.rotationsVar.get()
        if method == METHOD_PHASH:
            threshold = int(self.phashBitsVar.get())
        else:
            threshold = round(self.thresholdVar.get(), 2)

        def work(onProgress, onStatus):
            return findSimilarGroups(srcDir, method, modelKey, threshold, rotations,
                                     onProgress=onProgress, onStatus=onStatus,
                                     manualModelPath=manualModelPath)

        ScanWindow(self.root, self.onScanDone, work, 'Searching for similar photos in:', srcDir)

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
            self.showCurrentStart()
            modelPath = self._askForModelFile(payload)
            if modelPath:
                self.startScan(self.lastFolder, modelPath)
            else:
                self.showCurrentStart('Model not available - scan cancelled.')
            return
        if status == 'error':
            messagebox.showerror(APP_TITLE, f'Scan failed:\n{payload}', parent=self.root)
            self.showCurrentStart('Last scan failed.')
            return
        groups, unreadable, describe, providerName = payload
        self._warnUnreadable(unreadable)
        if not groups:
            messagebox.showinfo(APP_TITLE, 'No similar photos found.', parent=self.root)
            self.showCurrentStart(f'Last scan ({providerName}): no similar photos found.')
            return
        self._bringToFront()
        DuplicateReviewer(self.root, groups, self.onReviewDone, self.lastFolder,
                          self.deletedFolder, describe)

    def onReviewDone(self, deletedCount):
        messagebox.showinfo(APP_TITLE, f'Review finished.\nDeleted {deletedCount} file(s).',
                            parent=self.root)
        destination = self.deletedFolder or f'the {TRASH_NAME}'
        self.showCurrentStart(f'Last review: {deletedCount} file(s) moved to {destination}.')


if __name__ == '__main__':
    DedupeApp().run()
