import cv2
import mediapipe as mp
import pandas as pd
import easygui
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# ============ 多模態情緒檢測系統 ============

# 8類情緒定義
emotion_dict = {
    0: ("Neutral", "Neutral facial expression, no obvious emotional cues."),
    1: ("Happy", "The person's face is beaming with a bright, joyful smile."),
    2: ("Sad", "The person's brow is furrowed, mouth turned down, expressing sadness."),
    3: ("Surprised", "Eyebrows raised, eyes wide open, displaying a surprised expression."),
    4: ("Fearful", "The person's face is contorted with fear, eyes darting away nervously."),
    5: ("Angry", "Brow furrowed, eyes narrowed, face twisted in an angry expression."),
    6: ("Disgust", "Nose scrunched up, lips curled down, exhibiting a disgusted expression."),
    7: ("Contempt", "Face displays contempt with one-sided raised lip corner and slightly narrowed eyes.")
}

# ============ 資料前處理模組 ============

class DataPreprocessor:
    """資料前處理模組"""
    def __init__(self):
        # 圖像預處理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_3d_landmarks(self, landmarks, face_width, face_height):
        """3D人臉關鍵點前處理"""
        points_3d = []
        for landmark in landmarks.landmark:
            x = landmark.x * face_width
            y = landmark.y * face_height
            z = landmark.z * face_width
            points_3d.append([x, y, z])
        
        points_3d = np.array(points_3d, dtype=np.float32)
        
        # 正規化到[-1, 1]範圍
        points_3d[:, 0] = (points_3d[:, 0] / face_width) * 2 - 1
        points_3d[:, 1] = (points_3d[:, 1] / face_height) * 2 - 1
        points_3d[:, 2] = points_3d[:, 2] / face_width  # z已經是相對值
        
        return points_3d
    
    def preprocess_text_prompt(self, emotion_name):
        """文字提示語前處理"""
        # 創建情緒相關的文字描述
        prompts = {
            "Happy": "A person showing happiness with bright smile and joyful expression",
            "Sad": "A person displaying sadness with downturned mouth and furrowed brow",
            "Surprised": "A person expressing surprise with wide eyes and raised eyebrows",
            "Fearful": "A person showing fear with nervous expression and worried look",
            "Angry": "A person displaying anger with furrowed brow and tense expression",
            "Disgust": "A person showing disgust with scrunched nose and curled lips",
            "Contempt": "A person expressing contempt with one-sided lip raise",
            "Neutral": "A person with neutral facial expression and calm demeanor"
        }
        return prompts.get(emotion_name, prompts["Neutral"])
    
    def preprocess_face_image(self, face_crop):
        """人臉影像前處理"""
        # 轉換為PIL Image
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        # 應用變換
        face_tensor = self.image_transform(face_pil)
        return face_tensor

# ============ 特徵擷取模組 ============

class LandmarkEncoder(nn.Module):
    """3D關鍵點嵌入向量編碼器"""
    def __init__(self, num_landmarks=478, landmark_dim=3, embed_dim=256):
        super(LandmarkEncoder, self).__init__()
        input_dim = num_landmarks * landmark_dim
        
        # 添加更穩健的維度處理
        self.expected_input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, landmarks):
        batch_size = landmarks.shape[0]
        landmarks_flat = landmarks.view(batch_size, -1)
        
        # 動態調整輸入維度
        actual_dim = landmarks_flat.shape[1]
        if actual_dim != self.expected_input_dim:
            print(f"調整關鍵點維度: {actual_dim} -> {self.expected_input_dim}")
            if actual_dim < self.expected_input_dim:
                # 填充
                padding = torch.zeros(batch_size, self.expected_input_dim - actual_dim, 
                                    device=landmarks_flat.device, dtype=landmarks_flat.dtype)
                landmarks_flat = torch.cat([landmarks_flat, padding], dim=1)
            else:
                # 截取
                landmarks_flat = landmarks_flat[:, :self.expected_input_dim]
        
        return self.encoder(landmarks_flat)

class TextEncoder(nn.Module):
    """LLM文字嵌入向量編碼器"""
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=256):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)  # 雙向LSTM輸出 = hidden_dim * 2
        
        # 簡單的詞彙映射（實際應用中應使用預訓練的嵌入）
        self.word_to_idx = self._build_vocab()
    
    def _build_vocab(self):
        """建立簡單詞彙表"""
        words = ["person", "showing", "happiness", "sadness", "surprise", "fear", "anger", 
                "disgust", "contempt", "neutral", "smile", "expression", "eyes", "mouth",
                "bright", "joyful", "downturned", "furrowed", "wide", "raised", "nervous",
                "worried", "tense", "scrunched", "curled", "calm", "with", "and", "a"]
        return {word: idx for idx, word in enumerate(words)}
    
    def tokenize(self, text):
        """簡單分詞"""
        words = text.lower().split()
        tokens = [self.word_to_idx.get(word, 0) for word in words]
        return torch.tensor(tokens, dtype=torch.long)
    
    def forward(self, text_list):
        batch_embeddings = []
        for text in text_list:
            tokens = self.tokenize(text).unsqueeze(0)  # (1, seq_len)
            if tokens.size(1) == 0:  # 如果沒有識別的詞彙，添加一個padding token
                tokens = torch.zeros(1, 1, dtype=torch.long)
            
            embedded = self.embedding(tokens)  # (1, seq_len, embed_dim)
            lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: (2, 1, hidden_dim)
            
            # 使用最後時刻的輸出，而不是隱藏狀態
            # lstm_out的最後時刻: (1, hidden_dim*2)
            last_output = lstm_out[:, -1, :]  # 取最後一個時刻的輸出
            text_embed = self.fc(last_output)  # (1, embed_dim)
            batch_embeddings.append(text_embed.squeeze(0))  # 移除batch維度
        
        return torch.stack(batch_embeddings)  # (batch_size, embed_dim)

class ImageEncoder(nn.Module):
    """影像特徵嵌入向量編碼器"""
    def __init__(self, embed_dim=256):
        super(ImageEncoder, self).__init__()
        # 使用預訓練的ResNet
        self.backbone = models.resnet50(pretrained=True)
        # 移除最後的分類層
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 添加適配層
        self.adapter = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )
        
        # 凍結backbone的前幾層
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
    
    def forward(self, images):
        features = self.backbone(images)
        features = features.view(features.size(0), -1)
        return self.adapter(features)

# ============ 多模態特徵融合模組 ============

class MultimodalFusion(nn.Module):
    """多模態特徵融合模組"""
    def __init__(self, embed_dim=256):
        super(MultimodalFusion, self).__init__()
        self.embed_dim = embed_dim
        
        # 注意力機制
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # 融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        )
        
        # 特徵權重
        self.modal_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, landmark_features, text_features, image_features):
        # 堆疊所有特徵
        all_features = torch.stack([landmark_features, text_features, image_features], dim=1)
        
        # 自注意力機制
        attended_features, _ = self.attention(all_features, all_features, all_features)
        
        # 加權融合
        weighted_features = attended_features * self.modal_weights.unsqueeze(0).unsqueeze(-1)
        
        # 展平並通過融合層
        fused_features = weighted_features.view(weighted_features.size(0), -1)
        final_features = self.fusion_layer(fused_features)
        
        return final_features

# ============ 情緒變化檢測增強版 ============
class EmotionChangeTracker:
    """情緒變化追踪器"""
    def __init__(self, smoothing_window=5, change_threshold=0.3):
        self.emotion_history = []
        self.confidence_history = []
        self.smoothing_window = smoothing_window
        self.change_threshold = change_threshold
    
    def update(self, emotion_id, confidence):
        """更新情緒歷史"""
        self.emotion_history.append(emotion_id)
        self.confidence_history.append(confidence)
        
        # 保持固定窗口大小
        if len(self.emotion_history) > self.smoothing_window:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
    
    def get_smoothed_emotion(self):
        """獲取平滑後的情緒"""
        if not self.emotion_history:
            return 0, 0.0
        
        # 使用加權平均（更近的幀權重更高）
        weights = np.exp(np.linspace(0, 1, len(self.emotion_history)))
        weighted_emotions = []
        weighted_confidences = []
        
        for i, (emotion, conf) in enumerate(zip(self.emotion_history, self.confidence_history)):
            weighted_emotions.extend([emotion] * int(weights[i] * 10))
            weighted_confidences.append(conf * weights[i])
        
        # 選擇最常見的情緒
        if weighted_emotions:
            most_common_emotion = max(set(weighted_emotions), key=weighted_emotions.count)
            avg_confidence = np.mean(weighted_confidences)
            return most_common_emotion, avg_confidence
        
        return 0, 0.0
    
    def detect_change(self):
        """檢測情緒變化"""
        if len(self.emotion_history) < 2:
            return False, 0.0
        
        # 檢查最近的變化
        recent_emotions = self.emotion_history[-3:]  # 最近3幀
        unique_emotions = set(recent_emotions)
        
        if len(unique_emotions) > 1:
            # 計算變化強度
            change_intensity = len(unique_emotions) / len(recent_emotions)
            return change_intensity > self.change_threshold, change_intensity
        
        return False, 0.0

class EmotionChangeDetector(nn.Module):
    """情緒變化檢測模組"""
    def __init__(self, feature_dim=256, num_emotions=8):
        super(EmotionChangeDetector, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_emotions)
        )
        
        # 用於檢測變化的LSTM
        self.change_detector = nn.LSTM(feature_dim, 64, batch_first=True)
        self.change_classifier = nn.Linear(64, 2)  # 變化/無變化
    
    def forward(self, features, previous_features=None):
        # 情緒分類
        emotion_logits = self.classifier(features)
        
        # 如果有前一幀的特徵，檢測變化
        change_prob = None
        if previous_features is not None:
            # 計算特徵差異
            feature_diff = features - previous_features
            lstm_out, _ = self.change_detector(feature_diff.unsqueeze(1))
            change_logits = self.change_classifier(lstm_out[:, -1, :])
            change_prob = F.softmax(change_logits, dim=1)[:, 1]  # 變化的機率
        
        return emotion_logits, change_prob

# ============ 完整的多模態情緒檢測網絡 ============

class MultimodalEmotionNet(nn.Module):
    """完整的多模態情緒檢測網絡"""
    def __init__(self):
        super(MultimodalEmotionNet, self).__init__()
        self.landmark_encoder = LandmarkEncoder()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.multimodal_fusion = MultimodalFusion()
        self.emotion_detector = EmotionChangeDetector()
    
    def forward(self, landmarks, text_list, images, previous_features=None):
        # 特徵提取
        landmark_features = self.landmark_encoder(landmarks)
        text_features = self.text_encoder(text_list)
        image_features = self.image_encoder(images)
        
        # 多模態融合
        fused_features = self.multimodal_fusion(landmark_features, text_features, image_features)
        
        # 情緒檢測和變化分析
        emotion_logits, change_prob = self.emotion_detector(fused_features, previous_features)
        
        return emotion_logits, change_prob, fused_features

# ============ 主要處理函數 ============

def process_multimodal_emotion_detection():
    """多模態情緒檢測主函數"""
    
    # 選擇影片
    video_path = easygui.fileopenbox(
        msg="請選擇影片檔案",
        title="選擇影片",
        default="*.mp4",
        filetypes=["*.mp4", "*.avi", "*.mov"]
    )
    
    if not video_path:
        print("未選擇影片，程式結束")
        return
    
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建模型（注意：這是隨機初始化，實際應用需要預訓練權重）
    model = MultimodalEmotionNet().to(device)
    print("⚠️  注意: 目前使用隨機初始化模型，準確度有限")
    print("⚠️  建議: 載入預訓練權重或使用備用的關鍵點分類")
    print("✅ 已啟用改進的關鍵點分類作為備用方案")
    
    preprocessor = DataPreprocessor()
    
    # 準備輸出
    output_folder = os.path.splitext(video_path)[0] + "_multimodal_faces"
    os.makedirs(output_folder, exist_ok=True)
    excel_path = os.path.splitext(video_path)[0] + "_multimodal_emotions.xlsx"
    video_output_path = os.path.splitext(video_path)[0] + "_multimodal_processed.mp4"
    
    # MediaPipe初始化
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片：{video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 輸出影片設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    # 處理變數
    frame_idx = 0
    saved_idx = 0
    excel_data = []
    previous_features = None
    previous_emotion = None
    
    # 添加情緒變化追踪器
    emotion_tracker = EmotionChangeTracker(smoothing_window=5, change_threshold=0.4)
    
    print("開始多模態情緒檢測...")
    
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detection, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_det = face_detection.process(rgb_frame)
            
            if results_det.detections:
                for detection in results_det.detections:
                    # 獲取人臉邊界框
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x_min = max(0, int(bbox.xmin * w))
                    y_min = max(0, int(bbox.ymin * h))
                    box_w = min(w - x_min, int(bbox.width * w))
                    box_h = min(h - y_min, int(bbox.height * h))
                    
                    # 擷取人臉
                    face_crop = frame[y_min:y_min + box_h, x_min:x_min + box_w]
                    
                    # Face Mesh處理
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    mesh_results = face_mesh.process(face_rgb)
                    
                    if mesh_results.multi_face_landmarks:
                        landmarks = mesh_results.multi_face_landmarks[0]
                        fh, fw, _ = face_crop.shape
                        
                        try:
                            # 1. 資料前處理
                            landmarks_3d = preprocessor.preprocess_3d_landmarks(landmarks, fw, fh)
                            face_tensor = preprocessor.preprocess_face_image(face_crop).unsqueeze(0)
                            
                            # 2. 準備輸入
                            landmarks_tensor = torch.from_numpy(landmarks_3d).float().unsqueeze(0).to(device)
                            face_tensor = face_tensor.to(device)
                            
                            # 3. 模型推理
                            model.eval()
                            with torch.no_grad():
                                # 初始預測獲得可能的情緒（用於生成文字提示）
                                temp_emotion_logits, _, temp_features = model(
                                    landmarks_tensor, 
                                    ["neutral expression"],  # 臨時文字
                                    face_tensor,
                                    previous_features
                                )
                                temp_emotion_id = torch.argmax(temp_emotion_logits, dim=1).item()
                                temp_emotion_name = emotion_dict[temp_emotion_id][0]
                                
                                # 生成真正的文字提示
                                text_prompt = preprocessor.preprocess_text_prompt(temp_emotion_name)
                                
                                # 最終預測
                                emotion_logits, change_prob, features = model(
                                    landmarks_tensor,
                                    [text_prompt],
                                    face_tensor,
                                    previous_features
                                )
                                
                                # 獲得結果
                                probabilities = F.softmax(emotion_logits, dim=1)
                                emotion_id = torch.argmax(probabilities, dim=1).item()
                                confidence = torch.max(probabilities).item()
                                emotion_name, description = emotion_dict[emotion_id]
                                
                                # 變化檢測
                                change_score = change_prob.item() if change_prob is not None else 0.0
                                
                        except Exception as e:
                            print(f"多模態預測失敗: {e}")
                            emotion_id = 0
                            emotion_name = "Error"
                            description = f"Processing error: {str(e)}"
                            confidence = 0.0
                            change_score = 0.0
                            features = None
                    
                    else:
                        emotion_id = None
                        emotion_name = "No Face Mesh"
                        description = "No landmarks detected"
                        confidence = 0.0
                        change_score = 0.0
                        features = None
                    
                    # 更新情緒追踪器
                    emotion_tracker.update(emotion_id, confidence)
                    smoothed_emotion, smoothed_confidence = emotion_tracker.get_smoothed_emotion()
                    is_changing, change_intensity = emotion_tracker.detect_change()
                    
                    # 使用平滑後的情緒作為最終結果
                    final_emotion_id = smoothed_emotion
                    final_emotion_name = emotion_dict[final_emotion_id][0]
                    final_confidence = smoothed_confidence
                    
                    # 判斷情緒變化（基於追踪器）
                    if previous_emotion is None:
                        change_status = "-"
                    elif previous_emotion != final_emotion_id:
                        change_status = f"{previous_emotion}→{final_emotion_id}"
                    else:
                        change_status = "-"
                    
                    # 儲存人臉影像
                    save_name = f"face_{saved_idx:05d}.jpg"
                    save_path = os.path.join(output_folder, save_name)
                    cv2.imwrite(save_path, face_crop)
                    saved_idx += 1
                    
                    # 計算時間戳
                    timestamp_sec = frame_idx / fps
                    
                    # 記錄數據
                    excel_data.append({
                        "Frame": frame_idx,
                        "Timestamp (sec)": round(timestamp_sec, 2),
                        "X": x_min, "Y": y_min, "Width": box_w, "Height": box_h,
                        "Raw Emotion ID": emotion_id,
                        "Raw Emotion Name": emotion_name,
                        "Raw Confidence": round(confidence, 3),
                        "Smoothed Emotion ID": final_emotion_id,
                        "Smoothed Emotion Name": final_emotion_name,
                        "Smoothed Confidence": round(final_confidence, 3),
                        "Change Intensity": round(change_intensity, 3),
                        "Is Changing": is_changing,
                        "Change Status": change_status,
                        "Face Image": save_name
                    })
                    
                    # 繪製結果（使用平滑後的結果）
                    color = (0, 255, 0) if not is_changing else (0, 165, 255)  # 綠色正常，橙色變化
                    cv2.rectangle(frame, (x_min, y_min), (x_min + box_w, y_min + box_h), color, 2)
                    
                    text = f"{final_emotion_name} ({final_confidence:.2f})"
                    cv2.putText(frame, text, (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 如果檢測到變化，添加標記
                    if is_changing:
                        cv2.putText(frame, f"CHANGE! ({change_intensity:.2f})", 
                                   (x_min, y_min + box_h + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # 更新前一幀信息
                    previous_features = features
                    previous_emotion = final_emotion_id
            
            # 顯示和儲存
            cv2.imshow("Multimodal Emotion Detection", frame)
            out_video.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"已處理 {frame_idx} 幀")
    
    # 清理
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    
    # 輸出Excel
    print("正在輸出Excel檔案...")
    pd.DataFrame(excel_data).to_excel(excel_path, index=False)
    
    print(f"多模態情緒檢測完成！")
    print(f"臉部截圖: {output_folder}")
    print(f"分析結果: {excel_path}")
    print(f"處理影片: {video_output_path}")

# ============ 執行主程式 ============
if __name__ == "__main__":
    process_multimodal_emotion_detection()