# face_embedding_creator.py
import os
import json
import numpy as np
import pdb
from insightface.app import FaceAnalysis
import cv2

def create_face_embeddings(data_folder, output_embeddings_file, model_name='buufalo_s'):
    app = FaceAnalysis(name=model_name)
    app.prepare(ctx_id=0)
    face_embeddings = {}
    for person_name in os.listdir(data_folder):
        person_path = os.path.join(data_folder, person_name)
        if not os.path.isdir(person_path):
            continue

        person_embeddings = []
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            try:
                img = cv2.imread(image_path)
                faces = app.get(img)
                if faces:
                    embedding = faces[0].normed_embedding
                    person_embeddings.append(embedding.tolist())
                    print(f"Processed {person_name}: {image_name}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
        if person_embeddings:
            face_embeddings[person_name] = person_embeddings
    with open(output_embeddings_file, 'w') as f:
        json.dump(face_embeddings, f)
    
    print(f"Embeddings saved to {output_embeddings_file}")
    return face_embeddings

# def load_face_embeddings(embeddings_file):
#     """
#     Load face embeddings from a JSON file.
    
#     Args:
#         embeddings_file (str): Path to the embeddings JSON file
    
#     Returns:
#         dict: Dictionary of person names and their face embeddings
#     """
#     with open(embeddings_file, 'r') as f:
#         face_embeddings = json.load(f)
    
#     return face_embeddings

# def compare_face_embeddings(input_embedding, stored_embeddings, threshold=0.5):
#     """
#     Compare an input face embedding with stored embeddings.
    
#     Args:
#         input_embedding (np.array): Face embedding to compare
#         stored_embeddings (dict): Dictionary of stored face embeddings
#         threshold (float): Similarity threshold for face recognition
    
#     Returns:
#         list: Matching person names
#     """
#     from numpy.linalg import norm
    
#     matches = []
    
#     for person_name, person_embeddings in stored_embeddings.items():
#         for stored_embedding in person_embeddings:
#             # Calculate cosine similarity
#             cos_sim = np.dot(input_embedding, stored_embedding) / (norm(input_embedding) * norm(stored_embedding))
            
#             if cos_sim > threshold:
#                 matches.append(person_name)
#                 break  # Stop after first match for this person
    
#     return matches

# def recognize_face(input_img, known_embeddings, app):
#     # Perform face analysis on the input image
#     faces = app.get(input_img)
#     # Check if exactly one face is detected
#     if len(faces) != 1:
#         return "No face or multiple faces detected"
#     # Retrieve the embedding for the detected face
#     detected_embedding = faces[0].normed_embedding
#     # Calculate similarity scores with known embeddings
#     pdb.set_trace()
#     scores = np.dot(detected_embedding, np.array(known_embeddings).T)
#     scores = np.clip(scores, 0., 1.)
#     # Find the index with the highest score
#     idx = np.argmax(scores)
#     max_score = scores[idx]
    # Check if the maximum score is above a certain threshold (adjust as needed)
    # threshold = 0.7
    # if max_score >= threshold:
    #     recognized_name = names[idx]
    #     return f"Face recognized as {recognized_name} with a confidence score of {max_score:.2f}"
    # else:
    #     return "Face not recognized"

# Example usage
# if __name__ == '__main__':
#     # Create embeddings
#     data_folder = 'data'
#     output_file = 'face_embeddings.json'
#     create_face_embeddings(data_folder, output_file)
    
#     # Load embeddings (example)
#     loaded_embeddings = load_face_embeddings(output_file)
#     input_img = cv2.imread("image2.jpg")
#     input_img = cv2.imread("image.png")
#     faces = app.get(input_img)
#     input_embedding = faces[0].normed_embedding
#     # compare_face_embeddings(input_embedding, loaded_embeddings, threshold=0.5)
#     pdb.set_trace()
    # input_img = cv2.imread("image.png")
    # Call the recognize_face function
    # result = recognize_face(input_img, loaded_embeddings, app)
    # print(result)
