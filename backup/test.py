
import face_recognition
import os
print(f"face_recognition version: {face_recognition.__version__}")
print(f"face_recognition path: {os.path.dirname(face_recognition.__file__)}")
import dlib
print(f"dlib version is {dlib.__version__}")


# pip install --no-cache-dir --force-reinstall face-recognition==1.3.0