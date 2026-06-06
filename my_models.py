import cv2
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

def image_to_graph(image_tensor, k=9, patch_size=32, debug=False):
    C, H, W = image_tensor.shape

    if H < patch_size or W < patch_size:
        return None

    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    patches = patches.view(-1, C * patch_size * patch_size)

    if patches.size(0) < 2:
        return None

    similarity = cosine_similarity(patches.cpu().numpy())

    edge_index = []
    for i in range(len(patches)):
        indices = similarity[i].argsort()[-k-1:-1]
        edge_index += [(i, j) for j in indices]

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(
        x=patches.float(),
        edge_index=edge_index
    )

def extract_faces_from_video(video_path, max_faces=10):
    faces = []

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while cap.isOpened() and len(faces) < max_faces:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 10 == 0:
            frame = cv2.resize(frame, (128, 128))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces.append(frame)

    cap.release()

    return faces