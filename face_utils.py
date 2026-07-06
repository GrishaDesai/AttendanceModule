"""Face embedding extraction, matching, and liveness (blink) helpers."""
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector (crops + aligns to 160x160) and embedding model
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

mp_face_mesh = mp.solutions.face_mesh

# 6-point eye landmark indices (MediaPipe FaceMesh, 468-point model)
# Order: outer corner, top-outer, top-inner, inner corner, bottom-inner, bottom-outer
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def get_embedding(image_rgb: np.ndarray):
    """image_rgb: HxWx3 RGB numpy array. Returns a normalized 512-d embedding, or None if no face found."""
    face_tensor = mtcnn(image_rgb)
    if face_tensor is None:
        return None
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(device))
    emb = emb.cpu().numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def match_face(embedding, known_members, threshold: float = 0.65):
    """known_members: list of (member_id, name, embedding).
    Returns (member_id, name, score) for the best match above threshold, else None."""
    best_score = -1.0
    best_match = None
    for member_id, name, known_emb in known_members:
        score = cosine_similarity(embedding, known_emb)
        if score > best_score:
            best_score = score
            best_match = (member_id, name)
    if best_match is not None and best_score >= threshold:
        return best_match[0], best_match[1], best_score
    return None


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Standard 6-point Eye Aspect Ratio. Lower value = eye more closed."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices])
    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
    return ear