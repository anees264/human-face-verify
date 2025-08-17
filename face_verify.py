from deepface import DeepFace
import cv2
import numpy as np

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 RGB (H, W, 3)."""
    arr = img
    if arr.dtype in (np.float32, np.float64):
        maxv = arr.max() if arr.size else 1.0
        arr = np.clip(arr * (255.0 if maxv <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        raise ValueError("Unsupported image shape for face crop:", arr.shape)
    return arr

def _is_cartoonish_by_texture(gray: np.ndarray, variance_thresh: float = 60.0) -> bool:
    """
    Low Laplacian variance => flat, low-texture areas typical of cartoons.
    Tune variance_thresh higher for stricter filtering.
    """
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < variance_thresh

def _is_cartoonish_by_edges(gray: np.ndarray, edge_ratio_thresh: float = 0.02) -> bool:
    """
    Very low edge density is common in flat-shaded/animated faces.
    """
    edges = cv2.Canny(gray, 50, 150)
    ratio = (edges > 0).mean()
    return ratio < edge_ratio_thresh

def is_real_human_face(image_path: str,
                       detector_backend: str = "retinaface",
                       texture_var_thresh: float = 60.0,
                       edge_ratio_thresh: float = 0.02) -> bool:
    """
    Returns True only if a face is detected AND basic photo-realism checks pass.
    Rejects most animated/cartoon faces.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=True,
            detector_backend=detector_backend  # 'retinaface' is robust; you can try 'opencv' for fewer deps
        )
    except ValueError:
        print("❌ No face detected.")
        return False, "No face detected"

    if not faces:
        print("❌ No face detected.")
        return False, "No face detected"

    for idx, f in enumerate(faces, start=1):
        face_img = np.array(f["face"])
        face_rgb = _to_uint8_rgb(face_img)
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

        cartoon_by_texture = _is_cartoonish_by_texture(gray, variance_thresh=texture_var_thresh)
        cartoon_by_edges = _is_cartoonish_by_edges(gray, edge_ratio_thresh=edge_ratio_thresh)

        if cartoon_by_texture or cartoon_by_edges:
            print(f"⚠️ Face #{idx}: likely animated/cartoon (texture/edge heuristics failed).")
            return False, "Face is likely animated/cartoon"

    print(f"✅ Real human face detected ({len(faces)} face(s)).")
    return True, "Real human face detected"

if __name__ == "__main__":
    image_path = "test.jpg"
    is_real, reason = is_real_human_face(image_path)
    print(f"Is real: {is_real}, Reason: {reason}")
