from flask import Flask, render_template, request, redirect, url_for, session, send_file, Response, send_from_directory
import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = "super_secret_key"

known_faces_dir = "known_faces"
attendance_file = "attendance.csv"
os.makedirs(known_faces_dir, exist_ok=True)

def load_known_faces():
    faces, names = [], []
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(known_faces_dir, filename)
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if enc:
                faces.append(enc[0])
                names.append(os.path.splitext(filename)[0])
    return faces, names

known_faces, known_names = load_known_faces()


def mark_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(attendance_file):
        with open(attendance_file, "w") as f:
            f.write("Name,Time\n")

    try:
        df = pd.read_csv(attendance_file)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name", "Time"])

    if name not in df["Name"].values:
        df.loc[len(df)] = [name, now]
        df.to_csv(attendance_file, index=False)

ADMIN_USER = "admin"
ADMIN_PASS = "1234"


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            session["admin"] = True
            return redirect(url_for("admin_panel"))
        else:
            return render_template("login.html", error="nuv admin kadu ra pokiee🎀")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("login"))


@app.route('/known_faces/<path:filename>')
def serve_known_face(filename):
    return send_from_directory('known_faces', filename)


@app.route("/admin")
def admin_panel():
    if "admin" not in session:
        return redirect(url_for("login"))

    records = []
    if os.path.exists(attendance_file):
        try:
            df = pd.read_csv(attendance_file)
            if not df.empty:
                records = df.to_dict(orient="records")
        except pd.errors.EmptyDataError:
            with open(attendance_file, "w") as f:
                f.write("Name,Time\n")
            records = []
    else:
        with open(attendance_file, "w") as f:
            f.write("Name,Time\n")

    return render_template("admin.html", records=records)

@app.route("/delete_record", methods=["POST"])
def delete_record():
    if "admin" not in session:
        return redirect(url_for("login"))
    name = request.form["name"]
    df = pd.read_csv(attendance_file)
    df = df[df["Name"] != name]
    df.to_csv(attendance_file, index=False)
    return redirect(url_for("admin_panel"))

@app.route("/export")
def export_excel():
    if "admin" not in session:
        return redirect(url_for("login"))
    df = pd.read_csv(attendance_file)
    excel_path = "attendance.xlsx"
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

@app.route("/attendance")
def view_attendance():
    if "admin" not in session:
        return redirect(url_for("login"))

    if os.path.exists(attendance_file):
        try:
            df = pd.read_csv(attendance_file)
            records = df.to_dict(orient="records") if not df.empty else []
        except pd.errors.EmptyDataError:
            records = []
    else:
        records = []

    return render_template("attendance.html", records=records)

@app.route("/video_feed")
def video_feed():
    def generate():
        cam = cv2.VideoCapture(0)
        while True:
            success, frame = cam.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        cam.release()
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/add_face", methods=["GET", "POST"])
def add_face():
    if "admin" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form["name"].strip()
        if not name:
            return render_template("add_face.html", error="Please enter a name!")

        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cam.release()

        if not ret:
            return render_template("add_face.html", error="Failed to capture image.")

        path = os.path.join(known_faces_dir, f"{name}.jpg")
        cv2.imwrite(path, frame)

        global known_faces, known_names
        known_faces, known_names = load_known_faces()

        return render_template("add_face.html", success=True, name=name)

    return render_template("add_face.html")

@app.route("/video_capture")
def video_capture():
    def generate():
        cam = cv2.VideoCapture(0)
        while True:
            success, frame = cam.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        cam.release()
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/start_capture")
def start_capture():
    if "admin" not in session:
        return redirect(url_for("login"))

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return render_template("capture.html", error="Camera not accessible.")

    global known_faces, known_names
    if not known_faces:
        known_faces, known_names = load_known_faces()

    if len(known_faces) == 0:
        cam.release()
        return render_template("capture.html", error="No known faces found! Add faces first.")

    ret, frame = cam.read()
    cam.release()

    if not ret:
        return render_template("capture.html", error="Failed to capture image.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    marked = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_faces, face_encoding)
        if len(distances) == 0:
            continue

        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.55:
            name = known_names[best_match_index]
            mark_attendance(name)
            marked.append(name)

    if not marked:
        msg = "No known faces detected."
    else:
        msg = f"Attendance marked for: {', '.join(marked)}"

    return render_template("capture.html", success=msg)

@app.route("/capture")
def capture():
    if "admin" not in session:
        return redirect(url_for("login"))

    print("\nStarting live attendance capture...")

    global known_faces, known_names
    known_faces, known_names = load_known_faces()

    if len(known_faces) == 0:
        print("No known faces found! Add faces first.")
        return redirect(url_for("admin_panel"))

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Camera not accessible.")
        return redirect(url_for("admin_panel"))

    attendance_marked = set()
    start_time = datetime.now()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_faces, face_encoding)
            if len(distances) == 0:
                continue
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < 0.55:
                name = known_names[best_match_index]
                mark_attendance(name)
                attendance_marked.add(name)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Attendance Capture - Press Q to Stop", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if (datetime.now() - start_time).seconds > 10:
            break

    cam.release()
    cv2.destroyAllWindows()

    if attendance_marked:
        print(f"Attendance marked for: {', '.join(attendance_marked)}")
    else:
        print("No known faces recognized.")

    print("Capture session ended.\n")
    return redirect(url_for("admin_panel"))
@app.route("/capture_page")
def capture_page():
    if "admin" not in session:
        return redirect(url_for("login"))
    return render_template("capture.html")

if __name__ == "__main__":
    app.run(debug=True)