import cv2
import torch
from models.AutoToon import AutoToonModel  # Import từ models/AutoToon.py

# Load AutoToon model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/autotoon_model.pth"  # Đường dẫn đến model
autotoon = AutoToonModel(models_root="./models", force_cpu=(device.type == 'cpu'))
autotoon.load_model(epoch=0, save_dir="./checkpoints", device=device)  # Tải mô hình

# Cartoonize frame using AutoToon
def cartoonize_frame_with_autotoon(frame):
    """
    Apply AutoToon cartoonization to a frame.
    :param frame: A single video frame (numpy array).
    :return: Cartoonized frame.
    """
    # Resize frame về 256x256 (kích thước yêu cầu của mô hình)
    frame_resized = cv2.resize(frame, (256, 256))

    # Chuyển OpenCV frame (BGR) sang RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Chuyển frame sang tensor và normalize
    frame_tensor = torch.tensor(frame_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Xử lý frame qua mô hình
    with torch.no_grad():
        cartoonized_tensor, _, _ = autotoon(frame_tensor)

    # Chuyển kết quả từ tensor về numpy array (RGB)
    cartoonized_image = cartoonized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Chuyển từ RGB về BGR để hiển thị với OpenCV
    cartoonized_bgr = (cartoonized_image * 255).astype('uint8')
    return cv2.cvtColor(cartoonized_bgr, cv2.COLOR_RGB2BGR)

# Webcam feed
def cartoon_webcam_feed():
    """
    Capture video from webcam and apply cartoonization using AutoToon.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Cartoonize frame
        cartoon_frame = cartoonize_frame_with_autotoon(frame)

        # Hiển thị frame cartoonized
        cv2.imshow("Cartoonized Webcam Feed", cartoon_frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

# Chạy webcam feed
if _name_ == "_main_":
    cartoon_webcam_feed()