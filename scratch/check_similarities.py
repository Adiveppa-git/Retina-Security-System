import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
from app.services.face_recognition_service import extract_face_embedding, compute_face_similarity

imgs = [
    'project_images/003.jpeg',
    'project_images/004.jpeg',
    'project_images/005.jpeg',
    'project_images/009.jpeg',
    'project_images/014.jpeg',
    'project_images/015.jpeg'
]

embs = {}
for p in imgs:
    r = extract_face_embedding(p)
    if r['success']:
        embs[p] = r['embedding']
        print(f"{p}: conf={r['confidence']:.3f}")

print("\nPAIRWISE SIMILARITY MATRIX:")
names = [os.path.basename(p) for p in embs]
header = f"{'Image':12s} " + " ".join([f"{n:>10s}" for n in names])
print(header)
for p1 in embs:
    n1 = os.path.basename(p1)
    row = f"{n1:12s} "
    for p2 in embs:
        sim = compute_face_similarity(embs[p1], embs[p2])
        row += f"{sim:10.4f} "
    print(row)
