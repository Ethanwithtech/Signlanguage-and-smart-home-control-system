# 手语翻译与智能家居控制系统

这个应用程序使用计算机视觉技术来实现手语识别和手势控制智能家居设备。

## 功能特点

- **手语翻译**：识别美国手语(ASL)字母表手势并组成单词和句子
- **智能家居控制**：通过简单数字手势控制智能家居设备（灯光、窗帘、报警系统等）
- **双语界面**：支持中文和英文界面切换
- **骨架跟踪**：使用MediaPipe进行精确的手部骨架跟踪
- **AI翻译**：使用HKBU GenAI Platform API进行手语内容的翻译和理解
- **自定义API密钥**：支持在网页界面中设置自定义API密钥

## 系统要求

- **Python**：建议使用 Python 3.10 或 3.11（Python 3.12 可能与 TensorFlow 不兼容）
- **Visual C++ Build Tools**：在 Windows 上需要用于编译某些依赖包
- **摄像头**：用于捕获手势图像

## 安装指南

### 步骤1：环境准备

1. 安装 Python（推荐版本 3.10 或 3.11）
2. 在 Windows 上，安装 [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - 安装时选择"C++ 构建工具"工作负载

### 步骤2：安装依赖

使用启动脚本自动安装所有依赖项：

```bash
python run_app.py
```

或者手动安装依赖项：

```bash
pip install -r requirements.txt
```

### 步骤3：准备模型文件

确保手语识别模型文件已放置在正确位置。支持的模型文件名包括：
- `sign_language_model.h5`
- `sign_language_model_weights.h5`
- `best_sign_language_model.h5`

请将模型文件放在项目根目录或 `models` 文件夹中。

## 启动应用

运行启动脚本：

```bash
python combined_app.py
```

应用将在 http://localhost:5000 上启动，并自动打开浏览器。

## 使用指南

### 手语翻译

1. 打开"手语翻译"页面
2. 将手放在屏幕中央的绿色框内，做出ASL手语字母的手势
3. 系统会识别并显示对应的字母
4. 字母会组成单词和句子，可以用于翻译或理解
5. 头部向右或手掌向右添加空格
6. 头部向左或手掌向左结束句子并翻译

### 智能家居控制

1. 打开"智能家居控制"页面
2. 做出数字手势来控制设备：
   - 比数字1 = 开灯
   - 比数字2 = 关灯
   - 比数字3 = 开窗帘
   - 比数字4 = 关窗帘
   - 比数字5 = 激活报警系统
   - 握拳3秒钟 = 切换模式
   - 蜘蛛侠手势 = 激活报警系统(拇指、食指和小指伸出)

### API密钥配置

1. 在主页面的"API配置"部分可以设置自定义API密钥
2. 输入您的HKBU GenAI Platform API密钥并点击保存
3. 留空则使用系统默认API密钥
4. 自定义密钥将保存在浏览器本地存储中

## 常见问题解决

### 1. 安装依赖项失败

如果安装 pandas 或其他包失败，可能是因为缺少 C++ 编译器：

**解决方案**：安装 [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### 2. TensorFlow 导入错误

如果使用 Python 3.12，可能会出现 TensorFlow 兼容性问题：

**解决方案**：
- 降级到 Python 3.10 或 3.11
- 或安装特定版本的 TensorFlow：
  ```bash
  pip install tensorflow==2.12.0
  ```

### 3. MediaPipe 导入错误

**解决方案**：
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.0
```

### 4. 模型文件未找到或损坏

**解决方案**：
- 确保模型文件名正确且位于正确位置
- 可以尝试重新训练模型（参见 project.ipynb）

### 5. 摄像头未初始化

**解决方案**：
- 确保摄像头已连接并正常工作
- 检查是否有其他程序正在使用摄像头

## 系统架构

应用程序由以下主要组件组成：

1. **Flask Web 服务器**：处理 HTTP 请求并提供 Web 界面
2. **手语识别模型**：负责手语字母识别
3. **SmartHomeController**：管理智能家居设备状态
4. **手势检测系统**：使用 MediaPipe 进行手和头部姿态检测

### 主要组件功能

- **detect_sign_language(frame)**：处理手语翻译的视频帧和UI
- **detect_home_control(frame)**：处理智能家居控制的视频帧和UI
- **submit_to_ai(message)**：将手语内容发送到AI进行翻译
- **SmartHomeController**：控制智能家居设备状态

## API配置

本系统使用HKBU GenAI Platform API进行手语内容的翻译和理解。系统提供了两种配置API密钥的方式：

1. **默认API密钥**：系统内置默认密钥
2. **自定义API密钥**：用户可在主页的API配置部分设置自己的密钥

API相关设置：
- 基础URL：`https://genai.hkbu.edu.hk/general/rest`
- 模型名称：`gpt-4-o-mini`
- API版本：`2024-05-01-preview`

自定义API密钥将保存在浏览器本地存储中，并在会话期间使用。

## 手势检测与骨骼显示

本系统使用MediaPipe进行手势检测和骨骼显示。如果您发现手部骨骼未正确显示，请：

1. 确保在明亮条件下使用系统，避免复杂背景
2. 将手放在绿色ROI框中，保持动作稳定以获得最佳识别结果
3. 对于数字手势，确保手指清晰可见

## 系统改进历史

- **2024年4月版本**：
  - 将智能家居控制从复杂手势简化为简单数字手势(1-5)
  - 添加自定义API密钥功能
  - 分离手语翻译和智能家居控制界面
  - 优化UI设计和用户体验

## 贡献与开发

欢迎提交问题报告和功能请求。如果您想贡献代码，请遵循以下步骤：

1. Fork 仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 许可证

[MIT License](LICENSE)

## 项目说明
这是一个基于TensorFlow和OpenCV的手语识别系统，可以识别美国手语(ASL)字母，并支持智能家居控制功能。

## 解决模型加载问题

如果您遇到模型权重加载错误（`Layer count mismatch when loading weights from file`），请按照以下步骤解决：

1. **训练新模型**：运行以下命令来训练一个新的模型，该模型将创建兼容的权重文件：
   ```
   python model_trainer.py
   ```
   这将创建几个文件：
   - `best_sign_language_model.h5` - 训练中表现最佳的模型
   - `sign_language_model_final.h5` - 训练结束时的模型
   - `sign_language_model_weights.h5` - 兼容的权重文件
   - 还会更新原始权重文件位置的模型

2. **使用修改后的启动脚本**：
   ```
   python run_app.py
   ```
   这个脚本将：
   - 如果必要，创建一个新的模型
   - 设置正确的环境变量
   - 启动应用程序

3. **手动修复问题**：如果上述方法不起作用，您可以尝试：
   - 删除原始权重文件并使用新创建的模型
   - 修改 `app.py` 中的模型加载代码，使用正确的模型路径

## 开发者信息

如果需要修改或扩展系统功能：

1. 图像预处理在 `preprocess_for_prediction` 方法中实现
2. 手势检测逻辑在 `mediapipe_detect_gesture` 函数中
3. 为了调试，处理后的图像会保存在 `debug_images` 目录中





