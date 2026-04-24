# test_rtsp.py
import cv2

CAMERA_IP = "192.168.1.64"
USERNAME  = "admin"
PASSWORD  = "JABIN002@yunus"   # ✏️ put your actual camera password here

RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/102"

print(f"Connecting to camera...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Sub-stream (102) failed, trying main stream...")
    RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/101"
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Both streams failed — password wrong?")
else:
    print("✅ Connected! You should see camera feed now.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame read failed")
            break
        h, w = frame.shape[:2]
        cv2.putText(frame, f"HIKVision {w}x{h}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("HIKVision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()