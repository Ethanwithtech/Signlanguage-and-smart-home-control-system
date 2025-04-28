import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
import string
import requests
from flask import Flask, render_template, Response, jsonify
from PIL import Image, ImageDraw, ImageFont

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'signlanguage_secret_key'  # 添加Flask会话所需的密钥

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 设置OpenCV中文字体
def put_chinese_text(img, text, pos, color, font_size=30):
    """在图像上绘制中文文本，解决中文显示问号问题"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载系统字体
    font = None
    try:
        # 尝试加载微软雅黑字体（Windows系统常见字体）
        if os.path.exists("C:/Windows/Fonts/msyh.ttc"):
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        # 尝试加载黑体（MacOS系统常见字体）
        elif os.path.exists("/System/Library/Fonts/PingFang.ttc"):
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
        # 尝试Ubuntu字体
        elif os.path.exists("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size)
    except Exception as e:
        print(f"字体加载错误: {e}")
    
    # 绘制文本（如果没有字体，PIL会使用默认字体）
    draw.text(pos, text, font=font, fill=color)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

# 多语言支持函数
def translate_text(chinese_text, english_text):
    """根据当前系统设置返回中文或英文文本"""
    if use_pil_for_chinese:
        return chinese_text
    else:
        return english_text

try:
    import PIL
    from PIL import ImageFont
    use_pil_for_chinese = True
    # 检查是否能成功加载字体
    test_font = None
    try:
        if os.path.exists("C:/Windows/Fonts/msyh.ttc"):
            test_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
            print("✓ 成功加载微软雅黑字体")
        elif os.path.exists("/System/Library/Fonts/PingFang.ttc"):
            test_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
            print("✓ 成功加载PingFang字体")
        elif os.path.exists("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
            test_font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 24)
            print("✓ 成功加载文泉驿微米黑字体")
    except Exception as e:
        print(f"⚠ 字体加载警告: {e}")
        
    if test_font is None:
        print("⚠ 未找到中文字体，将使用默认字体，中文可能显示不正常")
    else:
        print("✓ PIL库已加载，将使用其处理中文显示")
except ImportError:
    use_pil_for_chinese = False
    print("⚠ PIL库未安装，中文可能显示为问号，建议安装PIL: pip install pillow")

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 增加为2只手，提高识别率
    min_detection_confidence=0.5,  # 降低检测阈值，提高灵敏度
    min_tracking_confidence=0.5
)

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize camera
camera = cv2.VideoCapture(0)

# HKBU GenAI Platform API configuration
default_apiKey = "06fd2422-8207-4a5b-8aaa-434415ed3a2b"
apiKey = default_apiKey  # 将使用这个变量，可以是默认或自定义的
basicUrl = "https://genai.hkbu.edu.hk/general/rest"
modelName = "gpt-4-o-mini"
apiVersion = "2024-05-01-preview"

# Load sign language model
# 使用相对路径加载模型，避免绝对路径问题
model_path = "best_sign_language_model.h5"
sign_model = None

print(f"尝试加载模型: {model_path}")
print(f"文件存在: {os.path.exists(model_path)}")

# 如果相对路径找不到，尝试绝对路径
if not os.path.exists(model_path):
    absolute_path = "C:/Users/dyc06/Desktop/signlanguage/best_sign_language_model.h5"
    if os.path.exists(absolute_path):
        model_path = absolute_path
        print(f"使用绝对路径: {model_path}")
        print(f"文件存在: {os.path.exists(model_path)}")

def load_model(model_path):
    """加载手语识别模型，尝试多种加载方法"""
    if os.path.exists(model_path):
        try:
            print("Loading model...")

            # 方法1：直接加载完整模型
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"✓ Method 1 success: Loaded complete model from {model_path}")
                return model
            except Exception as e1:
                print(f"Method 1 failed: {str(e1)}")

            # 方法2：创建模型架构并加载权重
            try:
                print("Trying method 2: Creating model architecture and loading weights")
                # 创建模型架构
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(26, activation='softmax')
                ])

                # 加载权重
                model.load_weights(model_path)
                print(f"✓ Method 2 success: Created model architecture and loaded weights from {model_path}")
                return model
            except Exception as e2:
                print(f"Method 2 failed: {str(e2)}")

            # 如果两种方法都失败，则抛出异常
            raise Exception("All loading methods failed")

        except Exception as e:
            print(f"❌ Model loading error: {e}")
            print(f"Error details: {str(e)}")

            # 创建一个简单的测试模型
            print("Creating a simple test model...")
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(26, activation='softmax')
            ])
            print("✓ Created a simple test model (Note: This is not a trained model)")
            return model
    else:
        print(f"❌ Model file not found: {model_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")

        # 尝试查找其他可能的模型文件
        directory = os.path.dirname(model_path)
        if os.path.exists(directory):
            print(f"Files in {directory}:")
            for file in os.listdir(directory):
                if file.endswith('.h5') or file.endswith('.weights'):
                    print(f"  - {file}")

        # 创建一个简单的测试模型
        print("Creating a simple test model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(26, activation='softmax')
        ])
        print("✓ Created a simple test model (Note: This is not a trained model)")
        return model

# 尝试加载模型
sign_model = load_model(model_path)
print(f"Model summary: {sign_model.summary()}")

# Initialize global variables
current_letter = ""
current_word = ""
sentence = ""
ai_response = ""
last_letter_time = time.time()
last_gesture_time = time.time()
last_confidence = 0.0  # 记录最后一次预测的置信度
mode_switch_time = time.time()  # 用于跟踪模式切换时间
current_mode = 0  # 0: 手语翻译模式, 1: 智能家居控制模式
space_added = False  # 标记是否已添加空格，避免重复添加
sentence_ended = False  # 标记是否已结束句子，避免重复翻译

# 添加更清晰的模式名称
MODE_SIGN_LANGUAGE = 0
MODE_SMART_HOME = 1

# ========== 新增：比数字手势识别映射 ==========
# 1-5分别对应家居操作
DIGIT_GESTURE_MAP = {
    1: ("开灯", "light_on"),
    2: ("关灯", "light_off"),
    3: ("开窗帘", "curtain_on"),
    4: ("关窗帘", "curtain_off"),
    5: ("报警", "alarm_on"),
}

# Smart Home Controller class
class SmartHomeController:
    def __init__(self):
        self.light_status = False
        self.curtain_status = False
        self.alarm_status = False

    def control(self, gesture_digit):
        """根据比数字手势直接控制智能家居设备"""
        if gesture_digit == 1:
            self.light_status = True
            return "灯光已打开"
        elif gesture_digit == 2:
            self.light_status = False
            return "灯光已关闭"
        elif gesture_digit == 3:
            self.curtain_status = True
            return "窗帘已打开"
        elif gesture_digit == 4:
            self.curtain_status = False
            return "窗帘已关闭"
        elif gesture_digit == 5:
            self.alarm_status = True
            return "报警系统已激活!!"
        else:
            return "未识别的数字手势"

    def reset(self):
        self.light_status = False
        self.curtain_status = False
        self.alarm_status = False
        return "所有设备已重置"

    def get_status(self):
        return {
            "light": "On" if self.light_status else "Off",
            "curtain": "Open" if self.curtain_status else "Closed",
            "alarm": "Active" if self.alarm_status else "Inactive"
        }

    def control_by_gesture(self, gesture_type):
        """根据特殊手势控制智能家居设备"""
        if gesture_type == GESTURE_OPEN_PALM:  # 张开手掌 - 全部打开
            self.light_status = True
            self.curtain_status = True
            return "所有设备已打开（灯光开启，窗帘打开）"
        elif gesture_type == GESTURE_FIST:  # 握拳 - 全部关闭
            self.light_status = False
            self.curtain_status = False
            self.alarm_status = False
            return "所有设备已关闭（灯光关闭，窗帘关闭，报警关闭）"
        elif gesture_type == GESTURE_HEART:  # 爱心手势 - 舒适模式
            self.light_status = True
            self.curtain_status = True
            self.alarm_status = False
            return "已切换到舒适模式（灯光开启，窗帘打开，报警关闭）"
        elif gesture_type == GESTURE_SPIDERMAN:  # 蜘蛛侠手势 - 触发报警
            self.alarm_status = True
            return "报警系统已激活!! 蜘蛛侠模式"
        else:
            return "未识别的特殊手势"

# Initialize Smart Home Controller
home_controller = SmartHomeController()

# Function to call AI API
def submit_to_ai(message):
    # 使用 Flask 会话中的 API Key 或默认值
    current_api_key = default_apiKey
    try:
        from flask import session
        if session and 'custom_api_key' in session:
            current_api_key = session['custom_api_key']
    except RuntimeError:
        # 不在请求上下文中，使用默认值
        pass
    
    print(f"\n=== 调用AI API ===")
    print(f"消息: {message}")
    print(f"使用API密钥: {'自定义密钥' if current_api_key != default_apiKey else '默认密钥'}")
    print(f"URL: {basicUrl}")

    # 构建对话内容
    conversation = [{"role": "user", "content": message}]
    url = basicUrl + "/deployments/" + modelName + "/chat/completions/?api-version=" + apiVersion
    headers = {'Content-Type': 'application/json', 'api-key': current_api_key}
    payload = {'messages': conversation}

    print(f"完整URL: {url}")
    print(f"请求头: {headers}")
    print(f"请求内容: {payload}")

    try:
        print("发送请求...")
        response = requests.post(url, json=payload, headers=headers)
        print(f"响应状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            result = data['choices'][0]['message']['content']
            print(f"AI响应: {result[:100]}..." if len(result) > 100 else f"AI响应: {result}")
            return result
        else:
            error_msg = f'错误: {response.status_code} - {response.text}'
            print(f"\u274c {error_msg}")
            
            # 如果API调用失败，返回默认翻译
            if "sign language" in message.lower():
                default_response = "抱歉，无法连接翻译服务。这可能是一段手语内容。"
                print(f"使用默认响应: {default_response}")
                return default_response
            return error_msg
    except Exception as e:
        error_msg = f'请求异常: {str(e)}'
        print(f"\u274c {error_msg}")
        
        # 如果API调用失败，返回默认翻译
        if "sign language" in message.lower() or "手语" in message:
            default_response = "抱歉，无法连接翻译服务。这可能是一段手语内容。"
            print(f"使用默认响应: {default_response}")
            return default_response
        return error_msg

# Preprocess image for model
def preprocess_image(image):
    """预处理图像以输入到模型，更好地适应训练数据特性"""
    try:
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用自适应阈值处理以突出手的轮廓
        # 这能更好地匹配训练数据的特性，训练数据都是二值化图像
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 应用轻微模糊以减少噪点
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        
        # 调整大小为模型输入尺寸 (28x28)
        resized = cv2.resize(blur, (28, 28))
        
        # 确保图像对比度适合模型 
        # MNIST手语数据集中的图像是白色手在黑色背景上
        if np.mean(resized) < 127:  # 如果平均亮度低于127，则反转
            resized = 255 - resized
            
        # 标准化像素值
        normalized = resized / 255.0
        
        # 重塑为模型输入格式
        input_data = normalized.reshape(1, 28, 28, 1)
        
        return input_data
    except Exception as e:
        print(f"图像预处理错误: {str(e)}")
        return None

# Predict letter from image
def predict_letter(image):
    global last_confidence
    
    if sign_model is None:
        print("\u26a0 警告: 模型未加载，无法预测字母")
        last_confidence = 0.0
        return "", 0.0
    
    if image is None or image.size == 0:
        print("\u26a0 警告: 图像为空，无法预测")
        last_confidence = 0.0
        return "", 0.0

    try:
        # 预处理图像
        input_data = preprocess_image(image)
        
        if input_data is None:
            return "", 0.0
            
        # 进行预测
        predictions = sign_model.predict(input_data, verbose=0)[0]

        # 获取预测结果
        letters = string.ascii_lowercase
        top_index = np.argmax(predictions)
        confidence = float(predictions[top_index])
        
        # 保存最后的置信度
        last_confidence = confidence

        # 记录高置信度预测
        if confidence > 0.7:
            print(f"预测字母: {letters[top_index]} 置信度: {confidence:.2f}")
            return letters[top_index], confidence
        else:
            # 如果置信度低，返回空字符串
            return "", confidence
    except Exception as e:
        print(f"\u274c 预测错误: {e}")
        print(f"错误详情: {str(e)}")
        last_confidence = 0.0
        return "", 0.0

# Detect hand gesture using MediaPipe
def detect_gesture(frame):
    global current_letter, current_word, sentence, ai_response, last_letter_time, last_gesture_time, mode_switch_time, current_mode, space_added, sentence_ended

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 创建一个干净的处理帧，不添加调试信息
    processed_frame = frame.copy()

    # Process with MediaPipe Holistic
    results = holistic.process(rgb_frame)

    # Variables for gesture detection
    gesture_digit = 0  # 新增：比数字手势编号
    gesture_text = "无手势"
    space_detected = False
    end_detected = False
    home_feedback = ""

    # 确保当前模式明确 - 0: 手语翻译模式, 1: 智能家居控制模式
    mode = current_mode  

    # 首先在界面顶部显示当前模式的大标题
    mode_title = "【手语翻译模式】" if mode == MODE_SIGN_LANGUAGE else "【智能家居控制模式】"
    
    # 使用背景高亮显示当前模式
    title_size = cv2.getTextSize(mode_title, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
    title_x = (processed_frame.shape[1] - title_size[0]) // 2
    
    # 添加背景矩形
    cv2.rectangle(processed_frame, 
                 (0, 0), 
                 (processed_frame.shape[1], 40),
                 (0, 70, 0) if mode == MODE_SIGN_LANGUAGE else (70, 0, 70), -1)
    
    # 显示模式标题
    if use_pil_for_chinese:
        processed_frame = put_chinese_text(processed_frame, mode_title, (title_x, 30), (255, 255, 255), font_size=30)
    else:
        mode_title_en = "【SIGN LANGUAGE MODE】" if mode == MODE_SIGN_LANGUAGE else "【SMART HOME CONTROL MODE】"
        cv2.putText(processed_frame, mode_title_en, (title_x, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    # 手语翻译模式和智能家居模式的UI分离
    if mode == MODE_SIGN_LANGUAGE:
        # 创建手语翻译区域
        height, width = processed_frame.shape[:2]
        roi_size = min(width, height) // 2
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2 + 50  # 下移以避开顶部标题

        # Draw ROI rectangle
        cv2.rectangle(processed_frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
        
        # 在ROI上方显示提示
        if use_pil_for_chinese:
            processed_frame = put_chinese_text(processed_frame, "请在此框内做手语动作", (roi_x, roi_y - 10), (0, 255, 255), font_size=24)
        else:
            cv2.putText(processed_frame, "Show sign language in this box", (roi_x, roi_y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Extract ROI
        roi = processed_frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
    else:
        # 智能家居模式，不需要ROI区域
        roi = None
        
        # 在屏幕中央显示手势控制提示
        if use_pil_for_chinese:
            processed_frame = put_chinese_text(processed_frame, "请做特定手势控制智能家居", (processed_frame.shape[1]//4, 80), (0, 255, 255), font_size=28)
        else:
            cv2.putText(processed_frame, "Make specific gestures to control smart home", (processed_frame.shape[1]//4, 80),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 添加模式切换提示
    if use_pil_for_chinese:
        processed_frame = put_chinese_text(processed_frame, "切换模式: 握拳并保持3秒", (10, processed_frame.shape[0] - 150), (255, 255, 0), font_size=20)
    else:
        cv2.putText(processed_frame, "Switch mode: Make a fist and hold for 3 seconds", (10, processed_frame.shape[0] - 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                   
    # 只在手语翻译模式下处理头部动作检测
    head_gesture = "无"
    if results.pose_landmarks and mode == MODE_SIGN_LANGUAGE:
        # Draw pose landmarks (简化标记点显示，只在需要时显示)
        mp_drawing.draw_landmarks(
            processed_frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=1),  # 减小圆圈和线条尺寸
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1)
        )
        
        # 获取头部关键点
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        left_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR]
        
        # 检测点是否有效
        if nose.visibility > 0.5 and left_ear.visibility > 0.5 and right_ear.visibility > 0.5:
            # 计算头部左右偏移
            head_center_x = (left_ear.x + right_ear.x) / 2
            head_offset = nose.x - head_center_x
            
            # 如果头部向右偏移，检测到空格
            if head_offset > 0.15:  # 头部明显向右
                head_gesture = "右偏"
                space_detected = True
                print("检测到头部右偏 - 空格手势")
            
            # 如果头部向左偏移，检测到句子结束
            elif head_offset < -0.15:  # 头部明显向左
                head_gesture = "左偏"
                end_detected = True
                print("检测到头部左偏 - 句子结束手势")
        
        # 在手语模式下显示头部手势
        if use_pil_for_chinese:
            processed_frame = put_chinese_text(processed_frame, f"头部动作: {head_gesture}", (10, 240), (255, 255, 0), font_size=20)
        else:
            head_gesture_en = "None" if head_gesture == "无" else "Right" if head_gesture == "右偏" else "Left"
            cv2.putText(processed_frame, f"Head gesture: {head_gesture_en}", (10, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 智能家居控制模式 - 总是显示设备状态
    if mode == MODE_SMART_HOME:
        # 显示当前设备状态
        status = home_controller.get_status()
        if use_pil_for_chinese:
            processed_frame = put_chinese_text(processed_frame, f"灯光: {'开启' if status['light'] == 'On' else '关闭'}", 
                                  (10, 180), (255, 255, 255), font_size=24)
            processed_frame = put_chinese_text(processed_frame, f"窗帘: {'打开' if status['curtain'] == 'Open' else '关闭'}", 
                                  (10, 210), (255, 255, 255), font_size=24)
            processed_frame = put_chinese_text(processed_frame, f"报警: {'激活' if status['alarm'] == 'Active' else '未激活'}", 
                                  (10, 240), (255, 255, 255), font_size=24)
        else:
            cv2.putText(processed_frame, f"Light: {'On' if status['light'] == 'On' else 'Off'}", (10, 180),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Curtain: {'Open' if status['curtain'] == 'Open' else 'Closed'}", (10, 210),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Alarm: {'Active' if status['alarm'] == 'Active' else 'Inactive'}", (10, 240),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 处理手部关键点检测 - 简化显示和增强手势识别
    if results.right_hand_landmarks:
        landmarks = results.right_hand_landmarks.landmark
        # 统计伸出手指数（不区分拇指）
        index_extended = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_extended = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_extended = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_extended = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        fingers = [index_extended, middle_extended, ring_extended, pinky_extended]
        num_extended = sum(fingers)
        # 五指全伸（含拇指）
        thumb_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x
        if num_extended == 4 and thumb_extended:
            gesture_digit = 5
        else:
            gesture_digit = num_extended
        if gesture_digit > 0:
            gesture_text = f"比{gesture_digit}"

    elif results.left_hand_landmarks:
        # 左手处理逻辑 (与右手类似)
        # 仅在智能家居模式下绘制详细的手部关键点，或在手语模式下简化显示
        if mode == MODE_SMART_HOME:
            mp_drawing.draw_landmarks(
                processed_frame,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
            )
        else:
            # 在手语模式下简化显示，只绘制轮廓
            mp_drawing.draw_landmarks(
                processed_frame,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1)
            )

        # 获取左手地标点
        landmarks = results.left_hand_landmarks.landmark
        
        # 基本左手关键点
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
        wrist = landmarks[mp_hands.HandLandmark.WRIST]

        # 检查手指是否伸展
        index_extended = index_tip.y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_extended = middle_tip.y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_extended = ring_tip.y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_extended = pinky_tip.y < landmarks[mp_hands.HandLandmark.PINKY_TIP].y

        # 获取拇指位置
        thumb_up = thumb_tip.y < wrist.y
        
        # 检查所有手指是否伸展
        all_extended = index_extended and middle_extended and ring_extended and pinky_extended
        
        # 检查所有手指是否弯曲
        all_closed = not index_extended and not middle_extended and not ring_extended and not pinky_extended

        # 分析当前模式下的手势
        if mode == MODE_SMART_HOME:
            # 智能家居控制模式的手势识别 - 左手镜像
            # 手势0：拇指向上，其他手指弯曲
            if thumb_up and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_digit = 0
                
            # 手势1：拇指向下，其他手指弯曲
            elif not thumb_up and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_digit = 1
                
            # 手势2：左手手掌向右（对应右手手掌向左）
            elif all_extended and thumb_tip.x > wrist.x:
                gesture_digit = 3  # 对应右手手势3，因为左右手镜像
                
            # 手势3：左手手掌向左（对应右手手掌向右）
            elif all_extended and thumb_tip.x < wrist.x:
                gesture_digit = 2  # 对应右手手势2，因为左右手镜像
                
            # 手势4：握拳
            elif all_closed:
                gesture_digit = 4

        # 智能家居控制处理
        if mode == MODE_SMART_HOME and gesture_digit > 0:
            home_feedback = home_controller.control(gesture_digit)

    # 根据当前模式显示合适的手势信息
    if mode == MODE_SMART_HOME:
        # 智能家居模式下显示手势ID及名称
        display_gesture = gesture_text if gesture_text != "无手势" else "等待手势..."
        
        if use_pil_for_chinese:
            processed_frame = put_chinese_text(processed_frame, f"当前手势: {display_gesture}", (10, 80), (0, 255, 0), font_size=24)
        else:
            gesture_text_en = display_gesture
            if display_gesture == "等待手势...":
                gesture_text_en = "Waiting for gesture..."
            elif "握拳" in display_gesture:
                gesture_text_en = "Fist"
                
            cv2.putText(processed_frame, f"Current gesture: {gesture_text_en}", (10, 80),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # 手语翻译模式下仅显示功能性手势（空格、句子结束等）
        if space_detected or end_detected or gesture_digit == 4:
            display_gesture = gesture_text
            
            if use_pil_for_chinese:
                processed_frame = put_chinese_text(processed_frame, f"功能手势: {display_gesture}", (10, 80), (0, 255, 0), font_size=24)
            else:
                gesture_text_en = "Palm right (Space)" if space_detected else "Palm left (End sentence)" if end_detected else "Fist (Hold to switch mode)"
                cv2.putText(processed_frame, f"Function gesture: {gesture_text_en}", (10, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 调试信息 - 显示当前检测到的手势ID (不管在哪种模式)
    if gesture_digit >= 0:
        debug_text = f"手势ID: {gesture_digit}" 
        if use_pil_for_chinese:
            processed_frame = put_chinese_text(processed_frame, debug_text, (10, 110), (255, 255, 0), font_size=20)
        else:
            cv2.putText(processed_frame, f"Gesture ID: {gesture_digit}", (10, 110),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 模式切换检测 - 握拳3秒
    current_time = time.time()
    if gesture_digit == 4 and current_time - mode_switch_time > 3.0:
        if mode == MODE_SIGN_LANGUAGE:
            current_mode = MODE_SMART_HOME
            print("模式切换: 手语翻译 -> 智能家居控制")
        else:
            current_mode = MODE_SIGN_LANGUAGE
            print("模式切换: 智能家居控制 -> 手语翻译")
        
        mode_switch_time = current_time
        
        # 在屏幕上显示模式切换信息
        switch_message = f"模式切换为: {'智能家居控制' if current_mode == MODE_SMART_HOME else '手语翻译'}"
        
        if use_pil_for_chinese:
            processed_frame = put_chinese_text(processed_frame, switch_message, 
                                   (processed_frame.shape[1] // 4, processed_frame.shape[0] // 2), (0, 0, 255), font_size=36)
        else:
            switch_message_en = f"Mode switched to: {'SMART HOME CONTROL' if current_mode == MODE_SMART_HOME else 'SIGN LANGUAGE'}"
            cv2.putText(processed_frame, switch_message_en, 
                      (processed_frame.shape[1] // 4, processed_frame.shape[0] // 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
    # 重置握拳超过3秒的计时，避免连续切换
    if gesture_digit != 4:
        mode_switch_time = current_time

    # 模式切换检测后，根据当前模式进行不同处理
    # 手语翻译处理
    if mode == MODE_SIGN_LANGUAGE:
        # 从ROI区域预测字母
        if roi is not None and roi.size > 0:
            letter, confidence = predict_letter(roi)

            # 当置信度足够高时更新当前字母
            current_time = time.time()
            if confidence > 0.7 and current_time - last_letter_time > 0.5:
                current_letter = letter
                last_letter_time = current_time
                print(f"预测字母: {letter} 置信度: {confidence:.2f}")
                
                # 在字母更新时重置空格和句子结束标记
                space_added = False
                sentence_ended = False

        # 处理空格手势 - 增加明显的视觉反馈
        current_time = time.time()
        if space_detected and current_time - last_gesture_time > 0.8 and not space_added:  # 缩短时间间隔为0.8秒
            if current_letter and current_letter != "":
                current_word += current_letter
                current_letter = ""
            
            # 只有当字符串不为空或不以空格结尾时才添加空格
            if current_word and not current_word.endswith(" "):
                current_word += " "  # 添加空格到当前单词
                space_added = True  # 标记已添加空格
                
                # 添加明显的视觉反馈
                if use_pil_for_chinese:
                    processed_frame = put_chinese_text(processed_frame, "✓ 已添加空格!", (processed_frame.shape[1] - 250, 50), (0, 255, 255), font_size=30)
                else:
                    cv2.putText(processed_frame, "✓ SPACE ADDED!", (processed_frame.shape[1] - 250, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
            last_gesture_time = current_time
            print(f"检测到空格手势, 当前单词: '{current_word}'")

        # 处理句子结束手势 - 增加明显的视觉反馈
        if end_detected and current_time - last_gesture_time > 0.8 and not sentence_ended:  # 缩短时间间隔为0.8秒
            if current_letter and current_letter != "":
                current_word += current_letter
                current_letter = ""

            if current_word:
                if sentence:
                    sentence += " " + current_word
                else:
                    sentence = current_word
                current_word = ""
                sentence_ended = True  # 标记句子已结束

                # 调用AI进行翻译
                print(f"发送到AI进行翻译: '{sentence}'")
                # 添加明显的视觉反馈
                if use_pil_for_chinese:
                    processed_frame = put_chinese_text(processed_frame, "✓ 句子结束! 正在翻译...", (processed_frame.shape[1] - 350, 100), (0, 255, 255), font_size=30)
                else:
                    cv2.putText(processed_frame, "✓ SENTENCE ENDED! TRANSLATING...", (processed_frame.shape[1] - 350, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                ai_response = submit_to_ai(f"翻译这段手语内容: {sentence}")
                print(f"AI响应: {ai_response}")

            last_gesture_time = current_time

        # 添加明确的空格和句子结束指示
        space_instruction = "⟹ 做手掌向右手势或头向右转动添加空格"
        end_instruction = "⟹ 做手掌向左手势或头向左转动结束句子"
        
        if use_pil_for_chinese:
            # 在显示区域附近添加指示
            processed_frame = put_chinese_text(processed_frame, space_instruction, (10, 270), (255, 255, 0), font_size=20)
            processed_frame = put_chinese_text(processed_frame, end_instruction, (10, 300), (255, 255, 0), font_size=20)
        else:
            cv2.putText(processed_frame, "=> Palm right or head right to add space", (10, 270),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(processed_frame, "=> Palm left or head left to end sentence", (10, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 显示手语翻译相关信息
        letter_confidence = "等待识别..."
        if current_letter and current_letter != "":
            letter_confidence = f"{current_letter.upper()} ({last_confidence:.2f})"
        
        if use_pil_for_chinese:
            # 不要在当前手势ID显示行显示，避免与功能手势冲突
            y_offset = 140  # 从140开始显示
            
            processed_frame = put_chinese_text(processed_frame, f"当前字母: {letter_confidence}", (10, y_offset), (0, 255, 0), font_size=24)
            y_offset += 30
            
            display_word = current_word if current_word else "等待输入..."
            display_sentence = sentence if sentence else "等待完整句子..."
            display_ai = ai_response if ai_response else "等待翻译..."
            
            # 添加引号，使显示的内容更清晰
            processed_frame = put_chinese_text(processed_frame, f"当前单词: '{display_word}'", (10, y_offset), (0, 255, 0), font_size=24)
            y_offset += 30
            
            processed_frame = put_chinese_text(processed_frame, f"完整句子: '{display_sentence}'", (10, y_offset), (0, 255, 0), font_size=24)
            y_offset += 30
            
            processed_frame = put_chinese_text(processed_frame, f"翻译结果: {display_ai[:50]+'...' if len(display_ai) > 50 else display_ai}", 
                                   (10, y_offset), (255, 255, 0), font_size=20)
            
            # 如果检测到空格或句子结束，在屏幕上显示提示
            if space_detected and current_time - last_gesture_time < 2.0:
                processed_frame = put_chinese_text(processed_frame, "检测到空格手势!", (processed_frame.shape[1] - 200, 50), (0, 255, 255), font_size=24)
                
            if end_detected and current_time - last_gesture_time < 2.0:
                processed_frame = put_chinese_text(processed_frame, "检测到句子结束手势!", (processed_frame.shape[1] - 300, 100), (0, 255, 255), font_size=24)
        else:
            y_offset = 140  # 从140开始显示
            
            cv2.putText(processed_frame, f"Current Letter: {letter_confidence}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            display_word = current_word if current_word else "Waiting..."
            display_sentence = sentence if sentence else "Waiting for sentence..."
            display_ai = ai_response if ai_response else "Waiting for translation..."
            
            # 添加引号，使显示的内容更清晰
            cv2.putText(processed_frame, f"Current Word: '{display_word}'", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            cv2.putText(processed_frame, f"Full Sentence: '{display_sentence}'", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            cv2.putText(processed_frame, f"Translation: {display_ai[:50]+'...' if len(display_ai) > 50 else display_ai}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 如果检测到空格或句子结束，在屏幕上显示提示
            if space_detected and current_time - last_gesture_time < 2.0:
                cv2.putText(processed_frame, "Space gesture detected!", (processed_frame.shape[1] - 250, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
            if end_detected and current_time - last_gesture_time < 2.0:
                cv2.putText(processed_frame, "End sentence gesture detected!", (processed_frame.shape[1] - 350, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 手语翻译模式下，头部动作控制空格/结束
    if mode == MODE_SIGN_LANGUAGE:
        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
            left_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
            right_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR]
            if nose.visibility > 0.5 and left_ear.visibility > 0.5 and right_ear.visibility > 0.5:
                head_center_x = (left_ear.x + right_ear.x) / 2
                head_offset = nose.x - head_center_x
                if head_offset > 0.15:
                    space_detected = True
                elif head_offset < -0.15:
                    end_detected = True

    # ======= 底部帮助说明 =======
    help_text = ""
    if mode == MODE_SMART_HOME:
        help_text = "请比数字1-5控制设备：1=开灯 2=关灯 3=开窗帘 4=关窗帘 5=报警"
    else:
        help_text = "将手放在绿色框内展示手语字母，头部向右或手掌向右=空格，头部向左或手掌向左=结束句子"
    # 在底部绘制背景条
    help_height = 40
    cv2.rectangle(processed_frame, (0, processed_frame.shape[0] - help_height), (processed_frame.shape[1], processed_frame.shape[0]), (0,0,0), -1)
    # 绘制帮助文字
    if use_pil_for_chinese:
        processed_frame = put_chinese_text(processed_frame, help_text, (10, processed_frame.shape[0] - help_height + 8), (255,255,0), font_size=22)
    else:
        cv2.putText(processed_frame, help_text, (10, processed_frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    return processed_frame

# ========== 新增：手势类型常量 ==========
# 常见手势定义
GESTURE_OPEN_PALM = "张开手掌"
GESTURE_FIST = "握拳"
GESTURE_HEART = "爱心手势"
GESTURE_NUMBER = "比数字"
GESTURE_SPIDERMAN = "蜘蛛侠手势"  # 新增蜘蛛侠手势类型

def detect_sign_language(frame):
    """只做手语翻译UI和识别，极简UI"""
    global current_letter, current_word, sentence, ai_response, last_letter_time, last_gesture_time, space_added, sentence_ended
    
    # 新增静态变量存储字母检测状态
    if not hasattr(detect_sign_language, "current_detecting_letter"):
        detect_sign_language.current_detecting_letter = ""
        detect_sign_language.letter_detect_start_time = 0
        detect_sign_language.sentence_ended = False
    
    # 创建干净的处理帧
    processed_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 处理MediaPipe
    results = holistic.process(rgb_frame)
    
    # 变量初始化
    head_nod_detected = False  # 点头 - 结束单词
    head_shake_detected = False  # 摇头 - 结束句子
    
    # 获取尺寸
    height, width = processed_frame.shape[:2]
    
    # 顶部模式标题
    mode_title = "【手语翻译模式】"
    title_size = cv2.getTextSize(mode_title, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
    title_x = (processed_frame.shape[1] - title_size[0]) // 2
    
    # 添加背景矩形
    cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], 40), (0, 70, 0), -1)
    
    # 显示模式标题
    if use_pil_for_chinese:
        processed_frame = put_chinese_text(processed_frame, mode_title, (title_x, 30), (255, 255, 255), font_size=30)
    else:
        cv2.putText(processed_frame, "【SIGN LANGUAGE MODE】", (title_x, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    # 创建手语翻译区域 (ROI)
    roi_size = min(width, height) // 2
    roi_x = (width - roi_size) // 2
    roi_y = (height - roi_size) // 2 + 50
    
    # 绘制ROI矩形
    cv2.rectangle(processed_frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
    
    # 处理双手检测 - 增强对左右手的支持
    hand_landmarks = None
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks
        mp_drawing.draw_landmarks(
            processed_frame,
            results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
        )
    
    if results.left_hand_landmarks:
        if not hand_landmarks:  # 如果右手未检测到，优先使用左手
            hand_landmarks = results.left_hand_landmarks
        mp_drawing.draw_landmarks(
            processed_frame,
            results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(117, 245, 66), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=1, circle_radius=1)
        )
    
    # 提取ROI - 使用更精确的区域提取
    roi = None
    if hand_landmarks:
        # 根据手的位置计算ROI边界框
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        
        # 添加边距
        padding = int(roi_size * 0.2)
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(width, max_x + padding)
        max_y = min(height, max_y + padding)
        
        # 确保区域是正方形且足够大
        box_size = max(max_x - min_x, max_y - min_y, 100)  # 最小100像素
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        # 计算新边界
        new_min_x = max(0, center_x - box_size // 2)
        new_min_y = max(0, center_y - box_size // 2)
        new_max_x = min(width, new_min_x + box_size)
        new_max_y = min(height, new_min_y + box_size)
        
        # 绘制基于手部的ROI（红色，表示实际提取区域）
        cv2.rectangle(processed_frame, (new_min_x, new_min_y), (new_max_x, new_max_y), (0, 0, 255), 2)
        
        # 提取手周围的ROI
        if new_min_x < new_max_x and new_min_y < new_max_y:
            hand_roi = frame[new_min_y:new_max_y, new_min_x:new_max_x]
            if hand_roi.size > 0:
                roi = hand_roi
    
    # 如果基于手部的ROI提取失败，则使用默认的矩形ROI
    if roi is None or roi.size == 0:
        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
    
    # 头部动作检测 - 改进点头和摇头检测
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            processed_frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1)
        )
        
        # 获取关键点
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        left_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR]
        
        if nose.visibility > 0.5 and left_ear.visibility > 0.5 and right_ear.visibility > 0.5:
            head_center_x = (left_ear.x + right_ear.x) / 2
            head_offset = nose.x - head_center_x
            
            # 点头检测 - 头部前后移动
            # 由于前后移动在2D图像中难以判断，我们基于头部尺寸变化检测
            # 当人点头时，头部在图像中的大小会变化
            head_width = abs(right_ear.x - left_ear.x)
            
            # 使用静态变量记录上一帧的头部宽度
            if not hasattr(detect_sign_language, "prev_head_width"):
                detect_sign_language.prev_head_width = head_width
            
            # 如果头部宽度变化超过阈值，视为点头 - 增大阈值减少误判
            head_width_change = abs(head_width - detect_sign_language.prev_head_width)
            if head_width_change > 0.05:  # 原来是0.02，现在增大为0.05
                head_nod_detected = True
                # 显示反馈
                if use_pil_for_chinese:
                    processed_frame = put_chinese_text(processed_frame, "检测到点头 - 结束单词", (width-350, 70), (0, 255, 255), font_size=24)
                else:
                    cv2.putText(processed_frame, "Nod detected - End word", (width-350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 摇头检测 - 头部左右移动
            if abs(head_offset) > 0.18:  # 原来是0.12，现在增大为0.18
                head_shake_detected = True
                # 显示反馈
                if use_pil_for_chinese:
                    processed_frame = put_chinese_text(processed_frame, "检测到摇头 - 结束句子", (width-350, 110), (0, 255, 255), font_size=24)
                else:
                    cv2.putText(processed_frame, "Head shake detected - End sentence", (width-350, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 更新上一帧的头部宽度
            detect_sign_language.prev_head_width = head_width

    # 从ROI预测手语字母
    if roi is not None and roi.size > 0:
        letter, confidence = predict_letter(roi)
        current_time = time.time()
        
        # 字母识别需保持2秒才确认为有效字母
        if confidence > 0.7:
            # 如果是新字母或者没有正在检测的字母
            if letter != detect_sign_language.current_detecting_letter:
                detect_sign_language.current_detecting_letter = letter
                detect_sign_language.letter_detect_start_time = current_time
                
                # 显示字母正在确认中
                if use_pil_for_chinese:
                    processed_frame = put_chinese_text(processed_frame, f"正在确认字母: {letter.upper()} ({confidence:.2f}) - 请保持手势", (roi_x, roi_y - 60), (255, 165, 0), font_size=24)
                else:
                    cv2.putText(processed_frame, f"Confirming: {letter.upper()} ({confidence:.2f}) - Hold gesture", (roi_x, roi_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # 如果同一字母持续检测超过2秒，确认为有效字母
            elif current_time - detect_sign_language.letter_detect_start_time >= 2.0:
                # 只有当字母真正变化时才更新
                if current_letter != letter:
                    current_letter = letter
                    last_letter_time = current_time
                    
                    # 立即将识别到的字母添加到当前单词中
                    current_word += current_letter
                    
                    # 在ROI上方显示识别到的字母
                    letter_display = f"识别字母: {current_letter.upper()} ({confidence:.2f})"
                    if use_pil_for_chinese:
                        processed_frame = put_chinese_text(processed_frame, letter_display, (roi_x, roi_y - 30), (0, 255, 0), font_size=28)
                        processed_frame = put_chinese_text(processed_frame, "✓ 字母已确认并添加到单词!", (roi_x + roi_size//2, roi_y - 60), (0, 255, 0), font_size=24)
                        processed_frame = put_chinese_text(processed_frame, f"当前单词: {current_word}", (10, height - 150), (255, 255, 0), font_size=24)
                    else:
                        cv2.putText(processed_frame, f"Letter: {current_letter.upper()} ({confidence:.2f})", (roi_x, roi_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(processed_frame, "✓ Letter confirmed and added to word!", (roi_x + roi_size//2, roi_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"Current word: {current_word}", (10, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # 重置当前字母，准备下一个字母的检测
                    current_letter = ""
                    detect_sign_language.current_detecting_letter = ""
            
            # 显示确认进度条
            else:
                progress = (current_time - detect_sign_language.letter_detect_start_time) / 2.0 * 100
                bar_width = int(roi_size * progress / 100)
                cv2.rectangle(processed_frame, (roi_x, roi_y - 50), (roi_x + bar_width, roi_y - 40), (0, 255, 0), -1)
                cv2.rectangle(processed_frame, (roi_x, roi_y - 50), (roi_x + roi_size, roi_y - 40), (255, 255, 255), 2)
                
                # 显示当前检测到的字母和确认进度
                if use_pil_for_chinese:
                    processed_frame = put_chinese_text(processed_frame, f"确认字母: {letter.upper()} ({confidence:.2f}) - {int(progress)}%", (roi_x, roi_y - 60), (255, 165, 0), font_size=24)
                else:
                    cv2.putText(processed_frame, f"Confirming: {letter.upper()} ({confidence:.2f}) - {int(progress)}%", (roi_x, roi_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            # 如果置信度低，重置检测状态
            detect_sign_language.current_detecting_letter = ""
    
    # 增强摇头检测 - 使用更明确的方式检测头部左右摆动
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        left_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR]
        
        if nose.visibility > 0.5 and left_ear.visibility > 0.5 and right_ear.visibility > 0.5:
            head_center_x = (left_ear.x + right_ear.x) / 2
            head_offset = nose.x - head_center_x
            
            # 保存历史头部位置用于检测摇头
            if not hasattr(detect_sign_language, "head_offset_history"):
                detect_sign_language.head_offset_history = []
            
            # 添加当前头部偏移到历史记录
            detect_sign_language.head_offset_history.append(head_offset)
            # 保持历史记录不超过10帧
            if len(detect_sign_language.head_offset_history) > 10:
                detect_sign_language.head_offset_history.pop(0)
            
            # 检测摇头 - 当头部左右摆动超过阈值且方向变化超过2次
            if len(detect_sign_language.head_offset_history) >= 5:
                direction_changes = 0
                for i in range(1, len(detect_sign_language.head_offset_history)):
                    prev = detect_sign_language.head_offset_history[i-1]
                    curr = detect_sign_language.head_offset_history[i]
                    # 如果方向从左到右或从右到左变化
                    if (prev < 0 and curr > 0) or (prev > 0 and curr < 0):
                        direction_changes += 1
                
                # 如果方向变化次数大于等于2，认为是摇头
                if direction_changes >= 2:
                    head_shake_detected = True
                    # 清空历史记录，避免连续触发
                    detect_sign_language.head_offset_history = []
                    
                    # 显示反馈
                    if use_pil_for_chinese:
                        processed_frame = put_chinese_text(processed_frame, "检测到摇头 - 结束句子", (width-350, 110), (0, 255, 255), font_size=24)
                    else:
                        cv2.putText(processed_frame, "Head shake detected - End sentence", (width-350, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 处理点头事件 - 结束当前单词
    current_time = time.time()
    if head_nod_detected and current_time - last_gesture_time > 1.0:
        # 将当前字母添加到单词中
        if current_letter and current_letter != "":
            current_word += current_letter
            current_letter = ""
            
        # 添加空格表示单词结束
        if current_word and not current_word.endswith(" "):
            current_word += " "
            
        # 更新时间戳和显示反馈
        last_gesture_time = current_time
        if use_pil_for_chinese:
            processed_frame = put_chinese_text(processed_frame, f"当前单词: {current_word}", (10, height - 150), (255, 255, 0), font_size=24)
        else:
            cv2.putText(processed_frame, f"Current word: {current_word}", (10, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 处理摇头事件 - 结束整个句子并翻译
    current_time = time.time()
    if head_shake_detected and current_time - last_gesture_time > 1.0 and not detect_sign_language.sentence_ended:
        # 将当前字母添加到单词中
        if current_letter and current_letter != "":
            current_word += current_letter
            current_letter = ""

        # 将当前单词添加到句子中
        if current_word:
            if sentence:
                sentence += " " + current_word.strip()
            else:
                sentence = current_word.strip()
            current_word = ""
            detect_sign_language.sentence_ended = True
            
            # 调用AI翻译
            if use_pil_for_chinese:
                processed_frame = put_chinese_text(processed_frame, "正在翻译...", (width//2 - 100, height//2), (0, 255, 255), font_size=36)
                # 更新显示来确保用户看到"正在翻译"的消息
                cv2.imshow("Sign Language", processed_frame)
                cv2.waitKey(1)
            
            ai_response = submit_to_ai(f"翻译这段手语内容: {sentence}")
            
            # 更新时间戳和显示反馈
            last_gesture_time = current_time
            if use_pil_for_chinese:
                processed_frame = put_chinese_text(processed_frame, f"句子完成: {sentence}", (10, height - 120), (255, 255, 0), font_size=24)
                processed_frame = put_chinese_text(processed_frame, f"翻译完成!", (10, height - 90), (255, 255, 0), font_size=24)
            else:
                cv2.putText(processed_frame, f"Sentence completed: {sentence}", (10, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(processed_frame, f"Translation completed!", (10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            # 添加明显的完成指示
            cv2.rectangle(processed_frame, (width//2 - 150, height//2 - 50), (width//2 + 150, height//2 + 50), (0, 255, 0), -1)
            if use_pil_for_chinese:
                processed_frame = put_chinese_text(processed_frame, "句子翻译完成!", (width//2 - 120, height//2 + 10), (255, 255, 255), font_size=30)
            else:
                cv2.putText(processed_frame, "Translation complete!", (width//2 - 140, height//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 移除句子重置功能
    # 如果用户再次摇头，重置状态以便开始新句子
    # if head_shake_detected and detect_sign_language.sentence_ended and current_time - last_gesture_time > 1.5:
    #     sentence = ""
    #     ai_response = ""
    #     detect_sign_language.sentence_ended = False
    #     last_gesture_time = current_time
    #     
    #     # 显示重置反馈
    #     cv2.rectangle(processed_frame, (width//2 - 150, height//2 - 50), (width//2 + 150, height//2 + 50), (0, 0, 255), -1)
    #     if use_pil_for_chinese:
    #         processed_frame = put_chinese_text(processed_frame, "句子已重置!", (width//2 - 100, height//2 + 10), (255, 255, 255), font_size=30)
    #     else:
    #         cv2.putText(processed_frame, "Sentence reset!", (width//2 - 100, height//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 显示当前识别进度
    if use_pil_for_chinese:
        # 当前状态信息显示
        display_letter = current_letter.upper() if current_letter else "_"
        display_word = current_word if current_word else "等待输入..."
        display_sentence = sentence if sentence else "等待完整句子..."
        
        processed_frame = put_chinese_text(processed_frame, f"当前字母: {display_letter}", (10, height - 210), (255, 255, 255), font_size=24)
        processed_frame = put_chinese_text(processed_frame, f"当前单词: {display_word}", (10, height - 180), (255, 255, 255), font_size=24)
        
        # 翻译结果显示
        display_ai = ai_response if ai_response else "等待翻译..."
        processed_frame = put_chinese_text(processed_frame, f"翻译结果: {display_ai}", (10, processed_frame.shape[0] - 90), (255, 255, 0), font_size=24)
    else:
        # 英文界面
        display_letter = current_letter.upper() if current_letter else "_"
        display_word = current_word if current_word else "Waiting for input..."
        display_sentence = sentence if sentence else "Waiting for sentence..."
        
        cv2.putText(processed_frame, f"Current letter: {display_letter}", (10, height - 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(processed_frame, f"Current word: {display_word}", (10, height - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 翻译结果显示
        display_ai = ai_response if ai_response else "Waiting for translation..."
        cv2.putText(processed_frame, f"Translation: {display_ai}", (10, processed_frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 底部帮助说明 - 修改为实际的数字手势控制
    help_text = translate_text(
        "手势控制: 1=开灯 2=关灯 3=开窗帘 4=关窗帘 5=报警 | 蜘蛛侠手势=报警系统 | 长按握拳=切换模式",
        "Gestures: 1=Light On 2=Light Off 3=Curtain Open 4=Curtain Close 5=Alarm | Spider-Man=Alarm | Hold Fist=Switch Mode"
    )
    help_height = 40
    cv2.rectangle(processed_frame, (0, processed_frame.shape[0] - help_height), (processed_frame.shape[1], processed_frame.shape[0]), (0, 0, 0), -1)
    
    if use_pil_for_chinese:
        processed_frame = put_chinese_text(processed_frame, help_text, (10, processed_frame.shape[0] - help_height + 8), (255, 255, 0), font_size=22)
    else:
        cv2.putText(processed_frame, translate_text(
            "1=开灯 2=关灯 3=开窗帘 4=关窗帘 5=报警 | 蜘蛛侠手势=报警 | 长按握拳=切换模式", 
            "1=Light On 2=Off 3=Curtain Open 4=Close 5=Alarm | Spider-Man=Alarm | Hold=Switch"), 
            (10, processed_frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return processed_frame

def detect_home_control(frame):
    """只做家居手势UI和识别，极简UI"""
    global last_gesture_time, mode_switch_time
    
    # 创建干净的处理帧
    processed_frame = frame.copy()
    
    # 处理MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)
    
    # 变量初始化
    gesture_type = None
    gesture_digit = 0
    home_feedback = ""
    
    # 获取尺寸
    height, width = processed_frame.shape[:2]
    
    # 设置识别区域（ROI）
    roi_size = min(width, height) // 2
    roi_x = (width - roi_size) // 2
    roi_y = (height - roi_size) // 2 + 50
    
    # 添加背景矩形
    cv2.rectangle(processed_frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 70, 0), 2)
    
    # 在ROI上方显示提示
    if use_pil_for_chinese:
        processed_frame = put_chinese_text(processed_frame, "请在此框内做家居手势", (roi_x, roi_y - 10), (0, 255, 255), font_size=24)
    else:
        cv2.putText(processed_frame, "Make specific gestures to control smart home", (roi_x, roi_y - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 处理双手手势检测 - 支持两只手
    hand_landmarks = None
    hand_type = ""
    
    # 先检查右手
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks
        hand_type = "right"
        mp_drawing.draw_landmarks(
            processed_frame,
            results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )
    
    # 再检查左手，如果右手未检测到
    if results.left_hand_landmarks and hand_landmarks is None:
        hand_landmarks = results.left_hand_landmarks
        hand_type = "left"
        mp_drawing.draw_landmarks(
            processed_frame,
            results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(117, 245, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=2, circle_radius=1)
        )
    
    # 如果检测到手，处理手势
    if hand_landmarks:
        landmarks = hand_landmarks.landmark
        
        # 检测手指是否伸展
        index_extended = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_extended = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_extended = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_extended = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        
        # 检测拇指伸展（考虑左右手差异）
        if hand_type == "right":
            thumb_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x
        else:  # 左手
            thumb_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x
        
        # 检测常见手势
        
        # 1. 张开手掌 - 五个手指都伸展（注意：这与数字5不同，主要用于功能性控制）
        if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            gesture_type = GESTURE_OPEN_PALM
            home_feedback = home_controller.control_by_gesture(GESTURE_OPEN_PALM)
        
        # 2. 握拳 - 所有手指收拢
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
            gesture_type = GESTURE_FIST
            home_feedback = home_controller.control_by_gesture(GESTURE_FIST)
        
        # 3. 心形手势 - 拇指和食指组成心形
        elif thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            # 检查拇指和食指指尖的距离是否很近
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            if distance < 0.1:  # 距离阈值，可能需要调整
                gesture_type = GESTURE_HEART
                home_feedback = home_controller.control_by_gesture(GESTURE_HEART)
                
        # 4. 蜘蛛侠手势 - 拇指、食指和小指伸出，中指和无名指弯曲
        elif thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
            gesture_type = GESTURE_SPIDERMAN
            home_feedback = home_controller.control_by_gesture(GESTURE_SPIDERMAN)
            
            # 显示蜘蛛侠手势反馈
            cv2.rectangle(processed_frame, (width//2 - 150, height//2 - 50), (width//2 + 150, height//2 + 50), (0, 0, 255), -1)
            if use_pil_for_chinese:
                processed_frame = put_chinese_text(processed_frame, "蜘蛛侠手势 - 报警已激活!", (width//2 - 140, height//2 + 10), (255, 255, 255), font_size=28)
            else:
                cv2.putText(processed_frame, "Spider-Man gesture - Alarm activated!", (width//2 - 140, height//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 5. 比数字 - 统计伸出的手指数量（不包括拇指）
        else:
            # 统计伸出的手指数量（不包括拇指）
            fingers = [index_extended, middle_extended, ring_extended, pinky_extended]
            num_extended = sum(fingers)
            
            # 比1-3 = 对应数量的手指伸展
            if 1 <= num_extended <= 3:
                gesture_digit = num_extended
                gesture_type = GESTURE_NUMBER
            
            # 比4 = 四指伸展但拇指不伸展
            elif num_extended == 4 and not thumb_extended:
                gesture_digit = 4
                gesture_type = GESTURE_NUMBER
            
            # 比5 = 四指伸展且拇指伸展（与张开手掌区分处理）
            elif num_extended == 4 and thumb_extended:
                gesture_digit = 5
                gesture_type = GESTURE_NUMBER
            
            # 显示检测到的手势调试信息
            if gesture_type == GESTURE_NUMBER and gesture_digit > 0:
                debug_text = f"检测到数字手势: {gesture_digit}"
                if use_pil_for_chinese:
                    processed_frame = put_chinese_text(processed_frame, debug_text, (10, height-200), (255, 165, 0), font_size=24)
                else:
                    cv2.putText(processed_frame, f"Detected digit: {gesture_digit}", (10, height-200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # 只在识别到比数字手势时控制家居
            if gesture_type == GESTURE_NUMBER and gesture_digit > 0:
                # 确保报警手势(5)能正确触发
                if gesture_digit == 5:
                    home_controller.alarm_status = True
                    home_feedback = "报警系统已激活!!"
                else:
                    home_feedback = home_controller.control(gesture_digit)
                last_gesture_time = time.time()
    
    # 显示设备状态
    status = home_controller.get_status()
    y = 70
    if use_pil_for_chinese:
        processed_frame = put_chinese_text(processed_frame, f"灯光: {'开启' if status['light']=='On' else '关闭'}  窗帘: {'打开' if status['curtain']=='Open' else '关闭'}  报警: {'激活' if status['alarm']=='Active' else '未激活'}", (10, y), (255, 255, 255), font_size=24)
        y += 40
        
        # 显示识别到的手势
        if gesture_type:
            if gesture_type == GESTURE_NUMBER:
                processed_frame = put_chinese_text(processed_frame, f"手势识别: {gesture_type} {gesture_digit}", (10, y), (0, 255, 255), font_size=24)
            else:
                processed_frame = put_chinese_text(processed_frame, f"手势识别: {gesture_type}", (10, y), (0, 255, 255), font_size=24)
            y += 40
            
            # 显示家居反馈
            if home_feedback:
                processed_frame = put_chinese_text(processed_frame, f"家居反馈: {home_feedback}", (10, y), (0, 255, 0), font_size=24)
    else:
        cv2.putText(processed_frame, f"Light: {status['light']} Curtain: {status['curtain']} Alarm: {status['alarm']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
        
        # 显示识别到的手势
        if gesture_type:
            if gesture_type == GESTURE_NUMBER:
                cv2.putText(processed_frame, f"Gesture: {gesture_type} {gesture_digit}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(processed_frame, f"Gesture: {gesture_type}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 40
            
            # 显示家居反馈
            if home_feedback:
                cv2.putText(processed_frame, f"Feedback: {home_feedback}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 底部帮助说明 - 修改为实际的数字手势控制
    help_text = translate_text(
        "手势控制: 1=开灯 2=关灯 3=开窗帘 4=关窗帘 5=报警 | 握拳两次=报警系统 | 长按握拳=切换模式 | 张开手掌=全开 | 爱心=舒适模式",
        "Gestures: 1=Light On 2=Light Off 3=Curtain Open 4=Curtain Close 5=Alarm | Double Fist=Alarm | Hold Fist=Switch Mode | Open Palm=All On | Heart=Comfort Mode"
    )
    help_height = 40
    cv2.rectangle(processed_frame, (0, processed_frame.shape[0] - help_height), (processed_frame.shape[1], processed_frame.shape[0]), (0, 0, 0), -1)
    
    if use_pil_for_chinese:
        processed_frame = put_chinese_text(processed_frame, help_text, (10, processed_frame.shape[0] - help_height + 8), (255, 255, 0), font_size=22)
    else:
        cv2.putText(processed_frame, translate_text(
            "1=开灯 2=关灯 3=开窗帘 4=关窗帘 5=报警 | 握拳两次=报警 | 长按握拳=切换模式", 
            "1=Light On 2=Off 3=Curtain Open 4=Close 5=Alarm | 2xFist=Alarm | Hold=Switch"), 
            (10, processed_frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return processed_frame

# ========== 新增：独立视频流生成 ==========
def generate_frames_sign():
    while True:
        success, frame = camera.read()
        if not success:
            break
        processed_frame = detect_sign_language(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_frames_home():
    while True:
        success, frame = camera.read()
        if not success:
            break
        processed_frame = detect_home_control(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ========== 路由调整 ==========
@app.route('/video_sign')
def video_sign():
    return Response(generate_frames_sign(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_home')
def video_home():
    return Response(generate_frames_home(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign_language')
def sign_language():
    return render_template('sign_language.html')

@app.route('/home_control')
def home_control():
    return render_template('home_control.html')

@app.route('/api/sign_status')
def sign_status():
    return jsonify({
        'current_letter': current_letter,
        'current_word': current_word,
        'sentence': sentence,
        'ai_response': ai_response
    })

@app.route('/api/home_status')
def home_status():
    return jsonify(home_controller.get_status())

@app.route('/api/reset_sign', methods=['POST'])
def reset_sign():
    global current_letter, current_word, sentence, ai_response
    current_letter = ""
    current_word = ""
    sentence = ""
    ai_response = ""
    return jsonify({'status': 'success'})

@app.route('/api/reset_home', methods=['POST'])
def reset_home():
    result = home_controller.reset()
    print(f"Reset smart home devices: {result}")
    return jsonify({'status': 'success', 'message': result})

@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    from flask import request, session
    data = request.get_json()
    
    if data and 'apiKey' in data:
        api_key = data['apiKey'].strip()
        
        if not api_key:
            # 如果API Key为空，则使用默认值
            session.pop('custom_api_key', None)
            return jsonify({
                'status': 'success',
                'message': '已重置为默认API密钥'
            })
        
        # 存储自定义API Key到会话中
        session['custom_api_key'] = api_key
        
        # 测试新API Key
        try:
            # 简单测试API连接
            test_message = "Hello, this is a test message to verify API connection."
            test_conversation = [{"role": "user", "content": test_message}]
            url = basicUrl + "/deployments/" + modelName + "/chat/completions/?api-version=" + apiVersion
            headers = {'Content-Type': 'application/json', 'api-key': api_key}
            payload = {'messages': test_conversation}
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return jsonify({
                    'status': 'success',
                    'message': 'API密钥验证成功并已保存'
                })
            else:
                # API Key无效，回退到默认值
                session.pop('custom_api_key', None)
                return jsonify({
                    'status': 'error',
                    'message': f'API密钥无效：{response.status_code} - {response.text}'
                })
        except Exception as e:
            # 连接错误，但仍保存API Key
            return jsonify({
                'status': 'warning',
                'message': f'API密钥已保存，但连接测试失败：{str(e)}'
            })
    
    return jsonify({
        'status': 'error',
        'message': '无效的请求数据'
    })

# Main function for standalone usage
def main():
    # Test API connection
    print("\n=== 测试API连接 ===")
    # 直接使用默认API进行测试
    print(f"使用默认API密钥进行测试")
    
    test_response = submit_to_ai("Hello, this is a test message to verify API connection.")
    print(f"API测试结果: {test_response[:100]}..." if len(test_response) > 100 else f"API测试结果: {test_response}")

    # Initialize camera
    if not camera.isOpened():
        print("错误: 无法打开摄像头")
        return

    print("摄像头成功打开")
    print("按 'ESC' 退出")
    print("按 'T' 测试API连接")
    print("按 'S' 模拟手语句子并翻译")
    print("\n=== 智能家居控制说明 ===")
    print("智能家居控制需要两个连续的手势:")
    print("  - 拇指向上 + 手掌向右: 打开灯光")
    print("  - 拇指向下 + 手掌向左: 关闭灯光")
    print("  - 手掌向右 + 拇指向上: 打开窗帘")
    print("  - 手掌向左 + 拇指向下: 关闭窗帘")
    print("  - 握拳 + 握拳: 激活报警系统")

    # Main loop
    while True:
        # Read frame
        ret, frame = camera.read()
        if not ret:
            print("错误: 无法读取画面")
            break

        # Process frame
        processed_frame = detect_gesture(frame)

        # Show frame
        cv2.imshow("手语和智能家居控制", processed_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('t'):  # Test API
            print("\n=== 手动API测试 ===")
            test_response = submit_to_ai("This is a manual test of the API connection.")
            print(f"API测试结果: {test_response}")
        elif key == ord('s'):  # Simulate sign language
            global sentence, ai_response
            print("\n=== 模拟手语翻译 ===")
            sentence = "hello how are you today"
            print(f"模拟的句子: {sentence}")
            ai_response = submit_to_ai(f"Translate this sign language content: {sentence}")
            print(f"AI响应: {ai_response}")

    # Release resources
    camera.release()
    cv2.destroyAllWindows()
    hands.close()
    holistic.close()

if __name__ == "__main__":
    # Check if running as standalone or as web app
    import sys
    import webbrowser
    import threading
    
    # 显示启动横幅
    print("\n" + "="*50)
    print("手语翻译与智能家居控制系统")
    print("="*50)
    
    # 显示模型加载状态
    if sign_model is not None:
        print(f"\n✓ 手语识别模型已加载")
    else:
        print(f"\n❌ 警告: 手语识别模型加载失败")
    
    # 测试API连接
    print("\n正在测试AI API连接...")
    # 直接使用默认API Key进行测试
    print(f"使用默认API密钥进行测试")
    
    try:
        test_response = submit_to_ai("Hello, this is a startup test message.")
        if "Error" in test_response or "错误" in test_response:
            print(f"❌ API连接测试失败: {test_response}")
        else:
            print(f"✓ API连接测试成功")
    except Exception as e:
        print(f"❌ API连接测试异常: {str(e)}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--standalone":
        main()
    else:
        print("=" * 50)
        print("启动网页服务器...")
        
        # 设置一个函数来在延迟后打开浏览器
        def open_browser():
            # 等待服务器启动
            time.sleep(2)
            # 打开浏览器
            webbrowser.open('http://127.0.0.1:5000/')
            print("✓ 已自动打开网站，如果浏览器没有打开，请手动访问 http://127.0.0.1:5000/")
        
        # 创建线程来打开浏览器
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True  # 设置为守护线程，这样当主程序退出时，线程会自动结束
        browser_thread.start()
        
        # 启动Flask应用
        print("✓ 服务器启动中，将自动打开浏览器...")
        app.run(debug=False)  # 设置debug=False防止自动重载干扰浏览器自动打开
