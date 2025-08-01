# 

# \# 🍎🍌 Fruit Ripeness Classification Using Deep Learning

# 

# This project leverages deep learning techniques to classify fruits into ripe and unripe categories using image data. The model is trained using PyTorch and exported in ONNX format. A Streamlit-based dashboard is developed for real-time inference and interaction.

# 

# ---

# 

# \## 📁 Project Structure

# 

# ```

# fruit-ripeness-classifier/

# │

# ├── archive.zip                # Raw dataset archive

# ├── dataset\_config.yaml        # Dataset configuration in YOLO-style format

# ├── fruit\_classifier.onnx      # Exported ONNX model

# ├── train\_model.ipynb          # Training notebook using PyTorch and ResNet18

# ├── streamlit\_app.py           # Streamlit dashboard for fruit classification

# └── README.md                  # Project documentation (you are here!)

# ```

# 

# ---

# 

# \## 🧠 Project Overview

# 

# \* \*\*Objective:\*\* Classify 22 categories of fruits (11 ripe + 11 unripe) using deep learning.

# \* \*\*Model:\*\* ResNet18 pretrained on ImageNet, fine-tuned on custom dataset.

# \* \*\*Deployment:\*\* ONNX model integrated with Streamlit for interactive UI.

# \* \*\*Dataset Size:\*\* Over 8,500+ labeled images.

# 

# ---

# 

# \## 📊 Dataset Details

# 

# \* \*\*Classes:\*\*

# 

# &nbsp; \* Ripe: Apple, Banana, Dragon Fruit, Grapes, Lemon, Mango, Orange, Papaya, Pineapple, Pomegranate, Strawberry

# &nbsp; \* Unripe: Same fruits as above in unripe form

# 

# \* \*\*Sample Class Distribution:\*\*

# 

# &nbsp; | Class Name       | Image Count |

# &nbsp; | ---------------- | ----------- |

# &nbsp; | Ripe Apple       | 388         |

# &nbsp; | Unripe Mango     | 400         |

# &nbsp; | Unripe Pineapple | 380         |

# &nbsp; | ...              | ...         |

# &nbsp; | \*\*Total\*\*        | \*\*8700+\*\*   |

# 

# \* \*\*Format:\*\* Folder-based structure with class-wise subdirectories

# 

# \* \*\*Config File:\*\* `dataset\_config.yaml` created for YOLO or training libraries

# 

# ---

# 

# \## 🧪 Model Training

# 

# \* \*\*Architecture:\*\* ResNet18 (transfer learning)

# \* \*\*Framework:\*\* PyTorch

# \* \*\*Loss Function:\*\* CrossEntropyLoss

# \* \*\*Optimizer:\*\* SGD (lr=0.001, momentum=0.9)

# \* \*\*Epochs:\*\* 10

# \* \*\*Accuracy:\*\* \\~64% on validation (baseline)

# 

# ✅ Code available in: `train\_model.ipynb`

# 

# ---

# 

# \## 🔁 Model Export

# 

# \* Format: `.onnx` (for portability)

# \* Exported using PyTorch’s `torch.onnx.export`

# \* Compatible with ONNX Runtime and web apps

# 

# ---

# 

# \## 🌐 Streamlit Dashboard

# 

# \* \*\*Framework:\*\* \[Streamlit](https://streamlit.io)

# \* \*\*Frontend Input:\*\* Image upload or webcam

# \* \*\*Backend:\*\* ONNX model inference with `onnxruntime`

# \* \*\*Real-Time Predictions:\*\* Displays class name and confidence

# \* \*\*Use:\*\* Simple web interface for testing and showcasing the model

# 

# ✅ Code available in: `streamlit\_app.py`

# 

# ---

# 

# \## 🚀 How to Run

# 

# \### 1. Clone the Repository

# 

# ```bash

# git clone https://github.com/yourusername/fruit-ripeness-classifier.git

# cd fruit-ripeness-classifier

# ```

# 

# \### 2. Install Dependencies

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \*Example `requirements.txt`:\*

# 

# ```

# torch

# torchvision

# onnx

# onnxruntime

# streamlit

# opencv-python

# Pillow

# pyyaml

# ```

# 

# \### 3. Run the Streamlit App

# 

# ```bash

# streamlit run streamlit\_app.py

# ```

# 

# ---

# 

# \## 📈 Future Enhancements

# 

# \* Improve accuracy with better data augmentation

# \* Deploy as a web API or mobile app

# \* Use quantized models for edge deployment

# \* Add object detection to detect and classify multiple fruits in an image

# 

# ---

# 

# \## ✍️ Author

# 

# \*\*Asghar Hussain\*\*

# AI Intern | Deep Learning Researcher | Software Engineering Student

# 📧 1212asghar@gmail.com

# 🔗 \\\[LinkedIn/GitHub Profile]

# 

# ---

# 

# \## 📄 License

# 

# This project is licensed under the MIT License. See \[LICENSE](LICENSE) for details.

# 



