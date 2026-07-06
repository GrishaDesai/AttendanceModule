"""
Academy Face Attendance App
- Tab 1: Register members (photo -> stored face embedding)
- Tab 2: Live webcam recognition with blink-based liveness check
"""
import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

from database import init_db, add_member, get_all_members, delete_member
from face_utils import get_embedding, match_face, eye_aspect_ratio, mp_face_mesh, LEFT_EYE, RIGHT_EYE

st.set_page_config(page_title="Academy Face Attendance", layout="wide")

conn = init_db()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

tab1, tab2 = st.tabs(["Add Member", "Live Recognition"])

# ---------------------------------------------------------------------------
# TAB 1: Add Member
# ---------------------------------------------------------------------------
with tab1:
    st.header("Add Academy Member")

    with st.form("add_member_form", clear_on_submit=True):
        member_id = st.text_input("Member ID")
        name = st.text_input("Name")
        photo_file = st.file_uploader("Upload Photo (clear, frontal, well-lit face)", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Add Member")

    if submitted:
        if not member_id or not name or not photo_file:
            st.error("Please fill all fields and upload a photo.")
        else:
            image = Image.open(photo_file).convert("RGB")
            image_np = np.array(image)
            embedding = get_embedding(image_np)
            if embedding is None:
                st.error("No face detected in the uploaded photo. Try a clearer, front-facing image.")
            else:
                add_member(conn, member_id.strip(), name.strip(), embedding)
                st.success(f"Added {name} ({member_id}) successfully.")

    st.divider()
    st.subheader("Registered Members")
    members = get_all_members(conn)
    if members:
        for mid, mname, _ in members:
            col1, col2, col3 = st.columns([2, 3, 1])
            col1.write(mid)
            col2.write(mname)
            if col3.button("Delete", key=f"del_{mid}"):
                delete_member(conn, mid)
                st.rerun()
    else:
        st.info("No members registered yet.")

# ---------------------------------------------------------------------------
# TAB 2: Live Recognition
# ---------------------------------------------------------------------------
with tab2:
    st.header("Live Face Recognition + Liveness Check")
    st.caption(
        "Blink naturally in front of the camera. Once a blink is detected, "
        "your face is matched against registered members."
    )

    known_members = get_all_members(conn)

    if not known_members:
        st.warning("No members registered yet. Add members in the first tab.")
    else:
        threshold = st.slider("Match confidence threshold", 0.40, 0.90, 0.65, 0.01)
        ear_threshold = st.slider("Blink sensitivity (EAR threshold, lower = stricter)", 0.15, 0.30, 0.21, 0.01)

        class Processor(VideoProcessorBase):
            def __init__(self):
                self.known_members = known_members
                self.threshold = threshold
                self.ear_threshold = ear_threshold
                self.consec_frames = 2       # frames eye must stay "closed" to count as a blink
                self.below_thresh_counter = 0
                self.blink_count = 0
                self.frame_counter = 0
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.last_match_label = ""

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                h, w, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)
                label = "No face detected"

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                    ear = (left_ear + right_ear) / 2.0

                    if ear < self.ear_threshold:
                        self.below_thresh_counter += 1
                    else:
                        if self.below_thresh_counter >= self.consec_frames:
                            self.blink_count += 1
                        self.below_thresh_counter = 0

                    self.frame_counter += 1
                    is_live = self.blink_count > 0

                    if not is_live:
                        label = "Please blink to verify liveness"
                    else:
                        # Only run the (expensive) embedding + match every 10 frames,
                        # reuse last result otherwise to keep the stream smooth.
                        if self.frame_counter % 10 == 0 or not self.last_match_label:
                            embedding = get_embedding(rgb)
                            if embedding is not None:
                                match = match_face(embedding, self.known_members, self.threshold)
                                if match:
                                    mid, mname, score = match
                                    self.last_match_label = f"{mname} ({mid}) {score:.2f}"
                                else:
                                    self.last_match_label = "Live - Unknown face"
                        label = self.last_match_label or "Verifying..."

                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="live-recognition",
            video_processor_factory=Processor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )