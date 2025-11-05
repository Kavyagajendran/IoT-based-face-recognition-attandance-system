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
import threading
import webbrowser
import subprocess
import platform
import numpy as np
import tkinter as tk
import locale

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

os_name = ""


def check_os():
    global os_name
    os_name = platform.system()
    if os_name == 'Windows':
        print("Operating system is Windows")
    elif os_name == 'Linux':
        print("Operating system is Linux")
    else:
        print("Unknown operating system")
    return os_name


check_os()
videoCaptureDeviceId = -1

LED_PIN = 18
led_on_flag = False
camera = None

frame_width = 640
frame_height = 480
'''
if os_name == 'Linux':
    frame_width = 320
    frame_height = 240
'''
print(f"Camera Frame Width={frame_width}\t Height={frame_height}")

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()
print(f'screen_width={screen_width}')
print(f'screen_height={screen_height}')

screen_x = (screen_width - frame_width) // 2
screen_y = (screen_height - frame_height) // 2


def get_webcams():
    port_ids = []
    global camera, videoCaptureDeviceId
    for port in range(2):
        print("Looking for a camera in port %s:" % port)
        camera = cv2.VideoCapture(port)
        if not camera.isOpened():
            print(f"Device {port} is not available")
            continue

        ret = camera.read()[0]
        if ret:
            backendName = camera.getBackendName()
            w = camera.get(3)
            h = camera.get(4)
            print("Camera %s (%s x %s) found in port %s " % (backendName, h, w, port))
            videoCaptureDeviceId = port
            port_ids.append(port)
            camera.release()
            break
        else:
            camera.release()
    return port_ids


get_webcams()

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
    # cv2.destroyWindow('MyWindow')  # Hides it

    if request.method == 'GET':
        led_off()
        return render_template('register.html')

    # import cv2
    global frame_width, frame_height, videoCaptureDeviceId, camera

    employee_name = request.form['employee_name'].strip()
    f_test(f"employee_name={employee_name}")

    # Validation
    if not employee_name or not all(c.isalnum() or c.isspace() for c in employee_name):
        flash('Invalid employee name. Only alphabets, numbers and spaces are allowed.', 'danger')
        return redirect(url_for('register'))

    if db.employee_exists(employee_name):
        flash(f'Employee "{employee_name}" already exists!', 'warning')
        return redirect(url_for('register'))

    f_test(f"employee name not exists")

    # Camera setup
    f_test(f"Capture varied angles")
    angle_instructions = [
        "Look straight", "Turn slightly left", "Turn slightly right",
        "Look up", "Look down", "Tilt head left",
        "Tilt head right", "Show left profile",
        "Show right profile", "Natural expression"
    ]

    camera = cv2.VideoCapture(videoCaptureDeviceId)
    time.sleep(0.5)  # Allow camera to warm up
    if not camera.isOpened():
        flash("Camera could not be opened!", 'danger')
        return redirect(url_for('register'))

    emp_dir = os.path.join(app.config['UPLOAD_FOLDER'], employee_name)
    os.makedirs(emp_dir, exist_ok=True)
    face_image_count = 10
    count = 0
    previous_frame = None
    frame_timeout_counter = 0
    MAX_UNCHANGED_FRAMES = 30
    duplicate_found = False
    timeout = 30
    start_time = time.time()
    matched_name = None

    # Force window creation
    window_name = 'Register New User'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    cv2.moveWindow(window_name, screen_x, screen_y)
    led_on()
    while count < face_image_count:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame")
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        display_frame = frame.copy()
        rgb_frame = frame[:, :, ::-1]

        try:
            f_test("getting face location")
            face_locations = face_recognition.face_locations(rgb_frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f_test("getting face location done")

            if face_locations:
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                f_test(f"Crop and save only the face region")
                face_image = frame[top:bottom, left:right]

                try:
                    f_test(f"Getting face encoding")
                    face_encoding = face_recognition.face_encodings(rgb_frame)[0]
                    f_test(f"Getting face encoding after")

                    matches = face_recognition.compare_faces(
                        face_utils.known_face_encodings,
                        face_encoding,
                        tolerance=0.5
                    )
                    f_test(f"matches")

                    if any(matches):
                        f_test(f"Duplicate found")
                        matched_index = matches.index(True)
                        matched_name = face_utils.known_face_names[matched_index]
                        duplicate_found = True
                        flash(f'This face already exists as "{matched_name}"! Registration cancelled.', 'danger')
                        break
                except Exception as e:
                    print(f"Encoding error: {e}")
                    continue

            # Add face count text
            cv2.putText(display_frame, f"Image {count + 1}/{face_image_count}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Check if window is still visible
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed manually")
                break

            # Show window
            try:
                f_test("Display Camera frame before")
                cv2.imshow(window_name, display_frame)
                f_test("Display Camera frame after")
            except cv2.error as e:
                print(f"cv2.imshow failed: {e}")
                break

            f_test(f"Verify face detected and Save only the face region")
            key = cv2.waitKey(100) & 0xFF
            if face_locations and (key == ord(' ') or key == 255 or key == -1):
                img_path = os.path.join(emp_dir, f"{count + 1:02d}.jpg")
                cv2.imwrite(img_path, face_image)

                image = face_recognition.load_image_file(img_path)
                if face_recognition.face_locations(image):
                    count += 1
                else:
                    os.remove(img_path)
        except Exception as e:
            print(f"Face detection error: {e}")
            continue

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()

    if duplicate_found:
        shutil.rmtree(emp_dir)
        led_off()
        return redirect(url_for('register'))

    if count < face_image_count:
        shutil.rmtree(emp_dir)
        led_off()
        flash(f'Failed to capture {face_image_count} valid face images', 'danger')
        return redirect(url_for('register'))

    try:
        f_test(f'Employee "{employee_name}" register')
        face_utils.train_new_face(employee_name)
        db.add_employee(employee_name)
        f_test("before flash")
        flash(f'Employee "{employee_name}" registered successfully!', 'success')
        f_test("after flash")
        blink_led(2)
    except Exception as e:
        print(f"Training failed: {e}")
        if os.path.exists(emp_dir):
            shutil.rmtree(emp_dir)
        flash(f'Training failed: {str(e)}', 'danger')

    led_off()
    return redirect(url_for('home'))


# @app.route('/register', methods=['GET', 'POST'])
def registerpart():
    flash(f'Not implemented', 'warning')
    return redirect(url_for('register'))
    f_test(f"request.method={request.method}")
    if request.method == 'POST':
        global frame_width, frame_height, videoCaptureDeviceId
        employee_name = request.form['employee_name'].strip()

        # Validation
        f_test(f"employee_name={employee_name}")
        if not employee_name or not all(c.isalnum() or c.isspace() for c in employee_name):
            flash('Invalid employee name. Only alphabets, numbers and spaces are allowed.', 'danger')
            return redirect(url_for('register'))

        if db.employee_exists(employee_name):
            flash(f'Employee "{employee_name}" already exists!', 'warning')
            return redirect(url_for('register'))

        f_test(f"employee name not exists")
        # Check for duplicate face with auto-detection
        camera = cv2.VideoCapture(videoCaptureDeviceId)

        if camera is None:
            camera = cv2.VideoCapture(videoCaptureDeviceId)
            # time.sleep(1)
        duplicate_found = False
        timeout = 30  # seconds
        start_time = time.time()
        matched_name = None
        cv2.destroyAllWindows()
        f_test(f"Empty Frame")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imshow('Face Registration', empty_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            f_test(f"quit")
        f_test(f"showing")
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                f_test(f"Failed to access webcam")
                flash('Failed to access webcam', 'danger')
                break
            frame = cv2.resize(frame, (frame_width, frame_height))
            f_test(f"Display instructions")
            # Display instructions
            cv2.putText(frame, "Position your face for registration", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Time remaining: {max(0, timeout - int(time.time() - start_time))}s",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert and detect faces
            rgb_frame = frame[:, :, ::-1]
            if rgb_frame.dtype != 'uint8':
                rgb_frame = rgb_frame.astype('uint8')
            # cv2.imshow('Face Registration', frame)
            f_test(f"Getting face locations")
            face_locations = face_recognition.face_locations(rgb_frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if face_locations:
                # Draw green box around face
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                try:
                    # Automatically check for duplicates
                    f_test(f"Getting face encoding")
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
            f_test(f"Display Face Registration frame")
            cv2.imshow('Face Registration', frame)
            f_test(f"Display Face Registration frame(b)")
            # Exit conditions

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (time.time() - start_time > timeout):
                break

        if duplicate_found:
            flash(f'This face already exists as "{matched_name}"! Registration cancelled.', 'danger')
            camera.release()
            cv2.destroyAllWindows()
            camera = None
            return redirect(url_for('register'))

        camera.release()
        cv2.destroyAllWindows()
        camera = None

        ls_ret = registerPart2(employee_name)
        f_test(ls_ret)
        if ls_ret == "redirect(url_for('home'))":

            return redirect(url_for('home'))
        elif ls_ret == "redirect(url_for('register'))":
            return redirect(url_for('register'))


    else:
        return render_template('register.html')


# @app.route('/attendance', methods=['GET', 'POST'])
@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    # cv2.imshow('Face Registration', display_frame)
    global frame_width, frame_height, videoCaptureDeviceId, camera
    # import cv2
    camera = cv2.VideoCapture(videoCaptureDeviceId)
    if camera is None:
        camera = cv2.VideoCapture(videoCaptureDeviceId)
        # time.sleep(1)
    start_time = time.time()
    global_timeout = 15  # Overall timeout if no faces detected
    multi_face_delay = 5  # Additional time when multiple faces detected
    processed_names = set()
    face_locations = None
    face_names = None
    unknown_count = 0
    window_name = 'Attendance Marking'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    cv2.moveWindow(window_name, screen_x, screen_y)
    led_on()
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            flash('Failed to capture image', 'danger')
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        # Face detection and recognition
        display_frame = frame.copy()
        rgb_frame = frame[:, :, ::-1]
        face_locations = None
        face_names = None
        try:

            face_locations, face_names = face_utils.recognize_faces(frame)

            if face_locations:
                # Draw annotations
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(display_frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                # Handle timing based on face count
                recognized_faces = [name for name in face_names if name != "Unknown"]
                num_faces = len(recognized_faces)

                unknown_count = face_names.count("Unknown")
                '''
                # Count and show unknown faces
                if unknown_count > 0:
                    cv2.putText(display_frame, f"Unknown: {unknown_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                '''
                # Single face - process immediately
                elapsed = time.time() - start_time
                if num_faces == 1 and unknown_count == 0:  # and (time.time() - start_time) > multi_face_delay:
                    break

                # Multiple faces - show for multi_face_delay seconds
                elif num_faces > 1 and unknown_count == 0 and elapsed > multi_face_delay:
                    break

            # Show countdown
            elapsed = time.time() - start_time
            remaining = max(0, int(global_timeout - elapsed))
            cv2.putText(display_frame, f"Time left: {remaining}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # Check if window is still visible
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed manually")
                break
            # Show window
            try:
                f_test("Display Camera frame before")
                cv2.imshow(window_name, display_frame)
                f_test("Display Camera frame after")
            except cv2.error as e:
                print(f"cv2.imshow failed: {e}")
                break

            # Exit conditions
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (elapsed > global_timeout):
                break
        except Exception as e:
            print(f"Face detection error: {str(e)}")  #
            continue

    camera.release()
    cv2.destroyAllWindows()

    # Process attendance and unknown faces
    recognized_names = [name for name in face_names if name != "Unknown"] if 'face_names' in locals() else []

    unknown_count = face_names.count("Unknown") #if 'face_names' in locals() else 0
    # print(f"face_names={face_names}")
    # print(f"unknown_count={unknown_count}")
    if unknown_count > 0:
        flash(f'Detected {unknown_count} unknown employee(s)', 'warning')

    if not recognized_names:
        flash('No recognized employees detected', 'warning')
        led_off()
        return redirect(url_for('home'))

    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today = date.today().strftime("%Y-%m-%d")
        att_count = 0
        for name in set(recognized_names):  # Process each employee once
            if name == "Unknown":
                continue

            # Skip if already processed in this session
            if name in processed_names:
                continue

            processed_names.add(name)
            has_checked_in = db.has_checked_in(name, today)
            has_checked_out = db.has_checked_out(name, today)

            if not has_checked_in:
                att_count = att_count + 1
                db.mark_attendance(name, current_time, None)
                flash(f'Checked in {name}', 'success')
            elif has_checked_in and not has_checked_out:
                att_count = att_count + 1
                db.update_checkout(name, today, current_time)
                flash(f'Checked out {name}', 'success')
            else:
                flash(f'{name} has already completed attendance today', 'info')
        if att_count > 0:
            blink_led(2)
        led_off()
        return redirect(url_for('home'))

    except Exception as e:
        flash(f'Error processing attendance: {str(e)}', 'danger')
        led_off()
        return redirect(url_for('home'))


def format_datetime(value, input_fmt, output_fmt):
    return datetime.strptime(value, input_fmt).strftime(output_fmt) if value else None


@app.route('/report', methods=['GET', 'POST'])
def attendance_report():
    led_off()
    locale.setlocale(locale.LC_TIME, '')
    today = datetime.today().strftime('%Y-%m-%d')
    from_date = datetime.today().date()
    to_date = datetime.today().date()
    if request.method == 'POST':
        if 'delete' in request.form:
            record_id = request.form['delete']
            db.delete_attendance_record(record_id)
            flash('Attendance record deleted successfully', 'success')
            return redirect(url_for('attendance_report'))

        from_date = request.form['from_date']
        to_date = request.form['to_date']


        records = db.get_attendance_between_dates(from_date, to_date)
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
    led_off()
    # employees = db.get_all_employees()
    employees = []
    for emp_id, name, reg_date in db.get_all_employees():
        # Get first image for each employee
        emp_dir = os.path.join(app.config['UPLOAD_FOLDER'], name)
        image_path = None
        if os.path.exists(emp_dir):
            image_files = [f for f in os.listdir(emp_dir) if f.endswith('.jpg')]
            if image_files:
                image_files = sorted([f for f in os.listdir(emp_dir) if f.endswith('.jpg')])
                # image_path = url_for('static', filename=f'employee_images/{name}/{image_files[0]}')
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


def is_browser_running():
    global os_name
    flag = False

    if os_name == 'Windows':
        import psutil
        browser_processes = ['chrome.exe', 'firefox.exe', 'iexplore.exe', 'MicrosoftEdge.exe']
        for process in psutil.process_iter(['name']):
            if process.info['name'] in browser_processes:
                flag = True
                break

    else:
        import subprocess
        output = subprocess.check_output(['ps', '-A'])
        flag = 'chromium' in output.decode() or 'chrome' in output.decode() or 'firefox' in output.decode() or 'safari' in output.decode()

    if flag:
        print("browser is running")
    else:
        print("browser is NOT running")
    return flag


def open_browser(url):
    url = "http://localhost:12000/"
    print("Opening browser")
    webbrowser.open(url)
    # print("after Opening browser")


def browser():
    url = "http://localhost:12000/"
    threading.Thread(target=open_browser, args=(url,)).start()


def close_browser():
    global os_name
    if os_name == 'Windows':
        return
        # subprocess.run('taskkill /F /IM chrome.exe', shell=True)
    elif os_name == 'Darwin':
        subprocess.run('pkill -f "Google Chrome"', shell=True)
    elif os_name == 'Linux':
        subprocess.run('pkill -f "chromium-browser"', shell=True)


def led_setup():
    global led_on_flag
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)  # GPIO Numbering of Pins
    GPIO.setup(LED_PIN, GPIO.OUT)  # Set LED_PIN as output

    led_off()
    led_on_flag = False


def led_on():
    global os_name, led_on_flag
    if os_name == 'Linux':
        global led_on_flag
        if led_on_flag == False:
            led_on_flag = True
            GPIO.output(LED_PIN, GPIO.HIGH)


def led_off():
    global os_name, led_on_flag
    if os_name == 'Linux':
        global led_on_flag
        if led_on_flag == True:
            led_on_flag = False
            GPIO.output(LED_PIN, GPIO.LOW)


def blink_led(duration=5, interval=0.125):
    global os_name
    if os_name == 'Linux':
        end_time = time.time() + duration

        while time.time() < end_time:
            led_on()  # LED On
            time.sleep(interval)  # Wait for the specified interval
            led_off()
            time.sleep(interval)  # Wait for the specified interval

        led_off()


def f_test(str):
    return
    # print(str)


if __name__ == '__main__':

    if os_name == 'Linux':
        import RPi.GPIO as GPIO  # RPi.GPIO can be referred as GPIO from now

        led_setup()
        blink_led()
    if is_browser_running():
        close_browser()
    browser()
    if os_name == 'Linux':
        app.run(port=12000, debug=False, threaded=False)
    else:
        app.run(port=12000, debug=True, threaded=False)

