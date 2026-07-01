"""人脸表情识别 Web 应用。支持上传图片和摄像头拍照识别。"""

import os

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from torchvision import transforms

from model import SimpleCNN

st.set_page_config(page_title="人脸表情识别", page_icon="😊", layout="wide")

EMOTIONS = [
    "愤怒",
    "厌恶",
    "恐惧",
    "开心",
    "悲伤",
    "惊讶",
    "中性",
]
EMOTION_EMOJIS = ["", "", "", "", "", "", ""]


class LiveEmotionState:
    def __init__(self):
        self.results = []
        self.error = None


LIVE_EMOTION_STATE = LiveEmotionState()


class EmotionVideoProcessor:
    def __init__(self, model, device, detector):
        self.model = model
        self.device = device
        self.detector = detector

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        try:
            result_frame, results = analyze_image(image, self.model, self.detector, self.device)
        except Exception as exc:
            LIVE_EMOTION_STATE.error = str(exc)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if results:
            LIVE_EMOTION_STATE.results = results
        else:
            LIVE_EMOTION_STATE.results = []

        processed_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(processed_frame, format="rgb24")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "checkpoints", "best_model.pth")

    if not os.path.exists(model_path):
        st.error(f"模型文件不存在: {model_path}")
        return None, device

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        st.error(f"模型文件加载失败: {e}")
        return None, device

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        st.error("模型权重格式不正确")
        return None, device

    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        st.error("人脸检测器加载失败")
        return None
    return detector


def load_chinese_font(font_size=16):
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None

FONT = load_chinese_font(16)


def get_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ]
    )


def predict_emotion(model, face_img, device):
    transform = get_transform()
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

    return idx, confidence, probs


def get_text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    if font is not None:
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        if hasattr(font, "getsize"):
            return font.getsize(text)
    return len(text) * 8, 16


def draw_result(frame, x, y, w, h, emotion_idx, confidence):
    if confidence > 0.8:
        color = (0, 255, 0)
    elif confidence > 0.5:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{EMOTIONS[emotion_idx]} {confidence * 100:.1f}%"

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = FONT
    text_width, text_height = get_text_size(draw, label, font)
    bg_x1 = x
    bg_y1 = y - text_height - 8
    if bg_y1 < 0:
        bg_y1 = 0
    bg_x2 = x + text_width + 8
    bg_y2 = bg_y1 + text_height + 8
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(color[2], color[1], color[0]))
    draw.text((x + 4, bg_y1 + 2), label, font=font, fill=(0, 0, 0))
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


def analyze_image(image, model, detector, device):
    if image is None:
        return None, []

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    if len(faces) == 0:
        return frame, []

    result_frame = frame.copy()
    results = []

    for i, (x, y, w, h) in enumerate(faces):
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue
        emotion_idx, confidence, probs = predict_emotion(model, face_roi, device)
        result_frame = draw_result(result_frame, x, y, w, h, emotion_idx, confidence)
        results.append(
            {
                "face_id": i + 1,
                "emotion": EMOTIONS[emotion_idx],
                "confidence": confidence,
                "probabilities": probs,
            }
        )

    return result_frame, results


def main():
    st.title("😊 人脸表情识别")
    st.caption("上传图片或使用摄像头拍照，快速识别人脸表情")

    model, device = load_model()
    face_detector = load_face_detector()

    if model is None or face_detector is None:
        return

    with st.sidebar:
        st.markdown("### 当前状态")
        st.success("模型已就绪")
        st.info(f"设备: {device}")

    tab_upload, tab_camera = st.tabs(["📤 上传图片", "📷 摄像头拍照"])

    with tab_upload:
        uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            result_frame, results = analyze_image(image, model, face_detector, device)

            if len(results) == 0:
                st.warning("未检测到人脸，请换一张更清晰的照片")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 原图")
                    st.image(np.array(image), use_container_width=True)
                with col2:
                    st.markdown("### 识别结果")
                    st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.markdown("### 识别详情")
                for result in results:
                    st.markdown(f"**{result['emotion']}** 置信度: {result['confidence'] * 100:.1f}%")
                    st.progress(result["confidence"])

    with tab_camera:
        st.markdown("### 实时识别")
        st.caption("允许浏览器摄像头权限后，画面会实时检测人脸并显示表情标签")

        col1, col2 = st.columns([2, 1])
        with col1:
            webrtc_streamer(
                key="emotion_camera",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=lambda: EmotionVideoProcessor(model, device, face_detector),
                async_processing=True,
            )

        with col2:
            st.markdown("### 当前结果")
            if LIVE_EMOTION_STATE.error:
                st.error(LIVE_EMOTION_STATE.error)
            elif not LIVE_EMOTION_STATE.results:
                st.info("等待摄像头画面...）")
            else:
                for result in LIVE_EMOTION_STATE.results:
                    st.markdown(f"**{result['emotion']}**")
                    st.progress(result["confidence"])
                    st.caption(f"置信度: {result['confidence'] * 100:.1f}%")


if __name__ == "__main__":
    main()
