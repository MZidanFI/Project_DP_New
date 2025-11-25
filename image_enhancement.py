import cv2
import numpy as np

def apply_he(img: np.ndarray) -> np.ndarray:
    """Histogram Equalization pada channel Y (Luminance)."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_contrast_stretching(img: np.ndarray) -> np.ndarray:
    """Linear Contrast Stretching per channel."""
    # Cara lebih efisien menggunakan cv2.normalize
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def apply_brightness(img: np.ndarray, alpha=1.2, beta=30) -> np.ndarray:
    """Mengatur Brightness (beta) dan Contrast (alpha)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_gamma(img: np.ndarray, gamma=1.5) -> np.ndarray:
    """Gamma Correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_unsharp(img: np.ndarray) -> np.ndarray:
    """Unsharp Masking untuk mempertajam tepi."""
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

def apply_bilateral(img: np.ndarray) -> np.ndarray:
    """Bilateral Filter (mengurangi noise tapi menjaga tepi)."""
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def apply_saturation(img: np.ndarray, scale=1.3) -> np.ndarray:
    """Meningkatkan saturasi warna."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

# --- FUNGSI UTAMA ---
def enhance_image(img: np.ndarray, method: str) -> np.ndarray:
    """
    Fungsi utama untuk memilih metode perbaikan citra.
    Menggunakan dictionary dispatch agar lebih rapi daripada if-else panjang.
    """
    if img is None:
        return None
        
    # Mapping nama metode ke fungsi
    processors = {
        "HE": apply_he,
        "CLAHE": apply_clahe,
        "CS": apply_contrast_stretching,
        "Brightness": apply_brightness,
        "Gamma": apply_gamma,
        "Unsharp": apply_unsharp,
        "Bilateral": apply_bilateral,
        "Saturation": apply_saturation
    }

    # Ambil fungsi berdasarkan nama, jika tidak ada/None, kembalikan gambar asli
    process_func = processors.get(method)
    
    if process_func:
        return process_func(img)
    
    return img