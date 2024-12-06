import cv2
from insightface.app import FaceAnalysis
import json
import numpy as np

class FaceDetector:
    def __init__(self, model_name):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0)
        self.loaded_embeddings = self.load_face_embeddings("face_embeddings.json")
    
    def load_face_embeddings(self, embeddings_file):
        with open(embeddings_file, 'r') as f:
            face_embeddings = json.load(f)
        return face_embeddings
    

    def compare_face_embeddings(self, input_embedding, stored_embeddings, threshold=0.5):
        from numpy.linalg import norm
        
        matches = []
        
        for person_name, person_embeddings in stored_embeddings.items():
            for stored_embedding in person_embeddings:
                cos_sim = np.dot(input_embedding, stored_embedding) / (norm(input_embedding) * norm(stored_embedding))
                
                matches.append({'name':person_name, 'score':round(float(cos_sim), 2)})
                # if cos_sim > threshold:
                    # break
        
        return matches

    def detect_faces(self, frame):
        faces = self.app.get(frame)
        annotated_img = frame.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            name = self.compare_face_embeddings(face.normed_embedding, self.loaded_embeddings, 0.5)
            if len(name) > 0:
                name = name[0]
            else:
                name = None
            name = str(name)
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            text_position = (bbox[0], bbox[1] - 10)
            cv2.putText(annotated_img, name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_img
