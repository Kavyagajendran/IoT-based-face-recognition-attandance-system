from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import cv2
import os
import shutil
import sqlite3
from datetime import datetime, date
from database import Database
from face_utils import FaceUtils
import face_recognition
import time
import pickle

app = Flask(__name__)
app.secret_key = '9751044317'  # The app.secret_key is used for security-related operations in a Flask application, primarily for Session, Flash message, Cookie Tampering
app.config['UPLOAD_FOLDER'] = 'employee_images'
app.config['KNOWN_FACES'] = 'known_faces'
app.config['DATABASE'] = 'attendance.db'

# Initialize components
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNOWN_FACES'], exist_ok=True)
face_utils = FaceUtils(app.config['UPLOAD_FOLDER'], app.config['KNOWN_FACES'])
db = Database(app.config['DATABASE'])

'''
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
'''


@app.route('/')
def home():
    today = date.today().strftime("%Y-%m-%d")
    attendance = db.get_todays_attendance(today)
    face_count = face_utils.get_face_count()
    for record in attendance:
        record["check_in"] = format_datetime(record["check_in"], "%Y-%m-%d %H:%M:%S", "%d-%b-%Y %I:%M:%S %p")
        record["check_out"] = format_datetime(record["check_out"], "%Y-%m-%d %H:%M:%S", "%d-%b-%Y %I:%M:%S %p")

    return render_template('home.html', attendance=attendance, face_count=face_count)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        employee_name = request.form['employee_name'].strip()

        # Validation
        if not employee_name or not all(c.isalnum() or c.isspace() for c in employee_name):
            flash('Invalid employee name. Only alphabets, numbers and spaces are allowed.', 'danger')
            return redirect(url_for('register'))

        if db.employee_exists(employee_name):
            flash(f'Employee "{employee_name}" already exists!', 'warning')
            return redirect(url_for('register'))

        # Check for duplicate face with auto-detection
        cap = cv2.VideoCapture(0)
        duplicate_found = False
        timeout = 30  # seconds
        start_time = time.time()
        matched_name = None

        while True:
            ret, frame = cap.read()
            if not ret:
                flash('Failed to access webcam', 'danger')
                break

            # Display instructions
            cv2.putText(frame, "Position your face for registration", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Time remaining: {max(0, timeout - int(time.time() - start_time))}s",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert and detect faces
            rgb_frame = frame[:, :, ::-1]
            if rgb_frame.dtype != 'uint8':
                rgb_frame = rgb_frame.astype('uint8')
            #cv2.imshow('Face Registration', frame)
            face_locations = face_recognition.face_locations(rgb_frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if face_locations:
                # Draw green box around face
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                try:
                    # Automatically check for duplicates
                    face_encoding = face_recognition.face_encodings(rgb_frame)[0]
                    # face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])[0]
                    '''
                    face_encoding = face_recognition.face_encodings(
                        rgb_frame,
                        known_face_locations=[face_locations[0]],
                        num_jitters=1
                    )[0]
                    '''
                    matches = face_recognition.compare_faces(
                        face_utils.known_face_encodings,
                        face_encoding,
                        tolerance=0.5
                    )

                    if any(matches):
                        matched_index = matches.index(True)
                        matched_name = face_utils.known_face_names[matched_index]
                        cv2.putText(frame, f"DUPLICATE: {matched_name}", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        duplicate_found = True
                        break  # Exit immediately if duplicate found

                    # If face detected and no duplicate, proceed to registration
                    break

                except Exception as e:
                    print(f"Face detection error: ")  # {str(e)}
                    continue

            # Show frame
            cv2.imshow('Face Registration', frame)

            # Exit conditions
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() - start_time > timeout):
                break

        if duplicate_found:
            flash(f'This face already exists as "{matched_name}"! Registration cancelled.', 'danger')
            return redirect(url_for('register'))

        # cap.release()
        cv2.destroyAllWindows()
        # Continue with registration if no duplicates and face detected

        # Capture varied angles
        angle_instructions = [
            "Look straight", "Turn slightly left", "Turn slightly right",
            "Look up", "Look down", "Tilt head left",
            "Tilt head right", "Show left profile",
            "Show right profile", "Natural expression"
        ]

        # cap = cv2.VideoCapture(0)
        emp_dir = os.path.join(app.config['UPLOAD_FOLDER'], employee_name)
        os.makedirs(emp_dir, exist_ok=True)
        face_image_count = 10
        count = 0
        while count < face_image_count:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for face detection
            rgb_frame = frame[:, :, ::-1]
            try:
                face_locations = face_recognition.face_locations(rgb_frame)

                # Create a copy for display with highlights
                display_frame = frame.copy()

                if face_locations:
                    # Get the first face found
                    top, right, bottom, left = face_locations[0]

                    # Draw green rectangle around face
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Crop and save only the face region
                    face_image = frame[top:bottom, left:right]

                # Display instructions
                cv2.putText(display_frame, f"Image {count + 1}/{face_image_count}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, angle_instructions[count], (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Register New User', display_frame)

                # Wait for 1s or space press
                key = cv2.waitKey(1000) & 0xFF
                if (key == ord(' ') or key == 255) and face_locations:  # Only capture if face found
                    img_path = os.path.join(emp_dir, f"{count + 1:02d}.jpg")

                    # Save only the face region
                    cv2.imwrite(img_path, face_image)
                    # Verify face detected
                    image = face_recognition.load_image_file(img_path)
                    if face_recognition.face_locations(image):
                        count += 1
                    else:
                        os.remove(img_path)  # Delete if no face
            except Exception as e:
                print(f"Face detection error: {str(e)}")  #
                continue

        cap.release()
        cv2.destroyAllWindows()

        if count < face_image_count:
            shutil.rmtree(emp_dir)
            flash(f'Failed to capture enough valid {face_image_count} face images', 'danger')
            return redirect(url_for('register'))

        try:
            face_utils.train_new_face(employee_name)
            db.add_employee(employee_name)
            flash(f'Employee "{employee_name}" registered successfully!', 'success')
        except Exception as e:
            if os.path.exists(emp_dir):
                shutil.rmtree(emp_dir)
            flash(f'Training failed: {str(e)}', 'danger')

        return redirect(url_for('home'))

    return render_template('register.html')


@app.route('/attendance', methods=['GET', 'POST'])
def mark_attendance():
    if request.method == 'POST':
        action = request.form['action']
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        timeout = 15
        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                flash('Failed to capture image', 'danger')
                cap.release()
                return redirect(url_for('mark_attendance'))

            # Make copy for display
            display_frame = frame.copy()
            face_locations, face_names = face_utils.recognize_faces(frame)

            # Draw face boxes and names
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Green box for recognized, red for unknown
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                thickness = 2

                # Draw rectangle around face
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, thickness)

                # Draw label background
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

                # Put name text
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            # Show time remaining
            elapsed_time = time.time() - start_time
            remaining_time = max(0, int(timeout - elapsed_time))
            cv2.putText(display_frame, f"Time left: {remaining_time}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Show the frame with annotations
            cv2.imshow('Attendance Marking', display_frame)
            last_frame = frame  # Store the last frame with faces

            # Break loop if faces detected or timeout
            if face_names and 1 == 0:
                break
            if elapsed_time > timeout:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if not face_names:
            flash('No faces detected or recognized', 'warning')
            return redirect(url_for('mark_attendance'))

        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            today = date.today().strftime("%Y-%m-%d")
            unknown_count = face_names.count("Unknown")

            if unknown_count > 0:
                flash(f'Detected {unknown_count} unknown employee(s)', 'warning')

            for name in face_names:
                if name == "Unknown":
                    continue

                if action == 'checkin':
                    if not db.has_checked_in(name, today):
                        db.mark_attendance(name, current_time, None)
                        flash(f'Check-in recorded for {name}', 'success')
                    else:
                        flash(f'{name} has already checked in today', 'warning')
                elif action == 'checkout':
                    if db.has_checked_in(name, today) and not db.has_checked_out(name, today):
                        db.update_checkout(name, today, current_time)
                        flash(f'Check-out recorded for {name}', 'success')
                    elif not db.has_checked_in(name, today):
                        flash(f'{name} has not checked in today', 'warning')
                    else:
                        flash(f'{name} has already checked out today', 'warning')

            return redirect(url_for('home'))

        except Exception as e:
            flash(f'Recognition error: {str(e)}', 'danger')
            return redirect(url_for('mark_attendance'))

    return render_template('attendance.html')


def format_datetime(value, input_fmt, output_fmt):
    return datetime.strptime(value, input_fmt).strftime(output_fmt) if value else None


@app.route('/report', methods=['GET', 'POST'])
def attendance_report():
    if request.method == 'POST':
        if 'delete' in request.form:
            record_id = request.form['delete']
            db.delete_attendance_record(record_id)
            flash('Attendance record deleted successfully', 'success')
            return redirect(url_for('attendance_report'))

        from_date = request.form['from_date']
        to_date = request.form['to_date']
        records = db.get_attendance_between_dates(from_date, to_date)
        today = datetime.today().strftime('%Y-%m-%d')

        from_date = datetime.strptime(from_date, "%Y-%m-%d")
        to_date = datetime.strptime(to_date, "%Y-%m-%d")
        formatted_from_date = from_date.strftime("%d-%b-%Y")
        formatted_to_date = to_date.strftime("%d-%b-%Y")
        # print(f"today={today}")
        for record in records:
            record["date"] = format_datetime(record["date"], "%Y-%m-%d", "%d-%b-%Y")
            record["check_in"] = format_datetime(record["check_in"], "%Y-%m-%d %H:%M:%S", "%I:%M:%S %p")
            record["check_out"] = format_datetime(record["check_out"], "%Y-%m-%d %H:%M:%S", "%I:%M:%S %p")
        return render_template('report.html', records=records, from_date=formatted_from_date, to_date=formatted_to_date,
                               today=today)
    today = datetime.today().strftime('%Y-%m-%d')
    return render_template('report.html', records=None, today=today)

@app.route('/employee_images/<employee_name>/<filename>')
def serve_employee_image(employee_name, filename):
    emp_dir = os.path.join(app.config['UPLOAD_FOLDER'], employee_name)
    return send_from_directory(emp_dir, filename)


@app.route('/manage_employees')
def manage_employees():
    #employees = db.get_all_employees()
    employees = []
    for emp_id, name, reg_date in db.get_all_employees():
        # Get first image for each employee
        emp_dir = os.path.join(app.config['UPLOAD_FOLDER'], name)
        image_path = None
        if os.path.exists(emp_dir):
            image_files = [f for f in os.listdir(emp_dir) if f.endswith('.jpg')]
            if image_files:
                image_files = sorted([f for f in os.listdir(emp_dir) if f.endswith('.jpg')])
                #image_path = url_for('static', filename=f'employee_images/{name}/{image_files[0]}')
                if len(image_files) >= 5:
                    image_path = url_for('serve_employee_image',
                                        employee_name=name,
                                        filename=image_files[4])  # 0-based index

        employees.append({
            'id': emp_id,
            'name': name,
            'registered_date': reg_date,
            'image_url': image_path
        })
    return render_template('manage_employees.html', employees=employees)


@app.route('/delete_employee', methods=['POST'])
def delete_employee():
    employee_id = request.form['employee_id']
    employee_name = request.form['employee_name']

    try:
        # 1. Delete from database
        db.delete_employee(employee_id)

        # 2. Delete image folder
        emp_dir = os.path.join(app.config['UPLOAD_FOLDER'], employee_name)
        if os.path.exists(emp_dir):
            shutil.rmtree(emp_dir)

        # 3. Check if any employees remain
        remaining_employees = db.get_all_employees()

        if not remaining_employees:
            # 4. Complete cleanup when no employees left
            encodings_file = os.path.join(app.config['KNOWN_FACES'], 'face_encodings.pkl')

            # Clear the encodings file
            with open(encodings_file, 'wb') as f:
                pickle.dump({'encodings': [], 'names': []}, f)

            # Clear the entire known_faces directory (optional)
            known_faces_dir = app.config['KNOWN_FACES']
            for filename in os.listdir(known_faces_dir):
                file_path = os.path.join(known_faces_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        else:
            # Standard encoding removal for single employee
            encodings_file = os.path.join(app.config['KNOWN_FACES'], 'face_encodings.pkl')

            if os.path.exists(encodings_file):
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)

                filtered_encodings = []
                filtered_names = []

                for encoding, name in zip(data['encodings'], data['names']):
                    if name != employee_name:
                        filtered_encodings.append(encoding)
                        filtered_names.append(name)

                with open(encodings_file, 'wb') as f:
                    pickle.dump({
                        'encodings': filtered_encodings,
                        'names': filtered_names
                    }, f)

        # 5. Update in-memory cache
        face_utils.known_face_encodings = []
        face_utils.known_face_names = []
        face_utils.load_known_faces()

        flash(f'Employee "{employee_name}" deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting employee: {str(e)}', 'danger')

    return redirect(url_for('manage_employees'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
