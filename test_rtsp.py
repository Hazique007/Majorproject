# test_rtsp.py
import cv2

RTSP_URL = "rtsp://admin:YOUR_PASSWORD@192.168.1.64:554/Streaming/Channels/102"

print("Connecting to HIKVision camera...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Failed to connect. Check:")
    print("   1. Is Ethernet cable plugged in?")
    print("   2. Did ping work? (ping 192.168.1.64)")
    print("   3. Is the username/password correct?")
    print("   4. Is camera IP correct?")
else:
    print("✅ Camera connected successfully!")
    print("   Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame read failed")
            break

        # Show resolution info on screen
        h, w = frame.shape[:2]
        cv2.putText(frame, f"HIKVision Stream: {w}x{h}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow("HIKVision Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()