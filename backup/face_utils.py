import os
import cv2
import pickle
import face_recognition
import numpy as np
#pip install face-recognition
class FaceUtils:
    def __init__(self, images_folder, known_faces_folder):
        self.images_folder = images_folder
        self.known_faces_folder = known_faces_folder
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        encodings_file = os.path.join(self.known_faces_folder, 'face_encodings.pkl')
        if os.path.exists(encodings_file):
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']

    def _load_existing_encodings(self):
        encodings_file = os.path.join(self.known_faces_folder, 'face_encodings.pkl')
        if os.path.exists(encodings_file):
            with open(encodings_file, 'rb') as f:
                return pickle.load(f)
        return {'encodings': [], 'names': []}

    def train_new_face(self, employee_name):
        emp_dir = os.path.join(self.images_folder, employee_name)
        existing_data = self._load_existing_encodings()
        
        new_encodings = []
        for img_file in sorted(os.listdir(emp_dir))[:5]:  # Use first 5 for speed
            img_path = os.path.join(emp_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            
            if face_locations:
                encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                new_encodings.append(encoding)
        
        if not new_encodings:
            raise ValueError(f"No valid faces found for {employee_name}")

        merged_data = {
            'encodings': existing_data['encodings'] + new_encodings,
            'names': existing_data['names'] + [employee_name] * len(new_encodings)
        }

        with open(os.path.join(self.known_faces_folder, 'face_encodings.pkl'), 'wb') as f:
            pickle.dump(merged_data, f)

        self.known_face_encodings = merged_data['encodings']
        self.known_face_names = merged_data['names']

    def recognize_faces(self, frame):
        if frame is None or frame.size == 0:  # Ensure frame is valid
            print("Error: Empty frame received.")
            return [], []
    
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:  # If no face is detected, return empty lists
            print("warning: No face is detected.")
            return [], []
        try:
            #print(f"Detected {len(face_locations)} faces.")
            '''        
            # Ensure `face_locations` is in correct format
            if not isinstance(face_locations, list) or not all(isinstance(loc, tuple) and len(loc) == 4 for loc in face_locations):
                print("Error: face_locations has an invalid format.")
                return [], []
            '''
            face_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                if self.known_face_encodings:  # Ensure known encodings exist
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)  # Get the best match

                    if face_distances[best_match_index] < 0.5:  # Check against the tolerance
                        name = self.known_face_names[best_match_index]
                    '''
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.5)
                    
                    
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    '''
                face_names.append(name)
        
            return face_locations, face_names
        except Exception as e:
            print(f"Error during face recognition: {e}")
            return [], []

    def retrain_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        for emp_name in os.listdir(self.images_folder):
            print(f"retrain_faces(), emp_name={emp_name}")
            emp_dir = os.path.join(self.images_folder, emp_name)
            if os.path.isdir(emp_dir):
                self.train_new_face(emp_name)

    def get_face_count(self):
        """Return count of unique employee faces"""
        return len(set(self.known_face_names))  # Using set to count unique names