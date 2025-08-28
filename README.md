# Explainable CNN-Based Alzheimer's MRI Classification with Streamlit Deployment

## 📌 Project Overview  
This project is part of the **Graduation Project** requirements.  
The aim is to develop a **deep learning pipeline** for classifying Alzheimer’s Disease stages using **MRI brain scans**. The model was trained using **Convolutional Neural Networks (CNNs)** and deployed as an **interactive Streamlit web application**.  

The classifier predicts the stage of Alzheimer’s and provides confidence scores.  


## 🧠 Dataset: Alzheimer’s Disease MRI  
- **Classes:**  
  1. Non Demented  
  2. Very Mild Demented  
  3. Mild Demented  
  4. Moderate Demented  

- **Data Composition:**  
  - Original MRI images.  
  - Augmented images (rotation, flipping, contrast adjustment).  

- **Split:**  
  - **70%** training  
  - **15%** validation  
  - **15%** testing  

📂 Dataset available on **[Kaggle](https://www.kaggle.com/datasets)**.  

---

## 🚀 Milestones Implemented  

### ✅ Milestone 1 — Data & Baseline CNN  
- Loaded dataset using `tf.keras.utils.image_dataset_from_directory`.  
- Computed **class weights** to handle class imbalance.  
- Built a **CNN model** (Sequential API).  
- Training protocol:  
  - `ModelCheckpoint` & `EarlyStopping`.  
  - Plotted learning curves (loss & accuracy).  
- Evaluation:  
  - Reported test accuracy & loss.  
  - Generated classification report & confusion matrix.  

### ✅ Milestone 2 — Streamlit Deployment  
Built an **interactive web app** that allows users to:  
- Upload an MRI image (`.jpg`, `.jpeg`, `.png`).  
- Get predicted class + confidence scores.  
- See class probabilities.  
- Display results in a clean, user-friendly UI.  

---

## ⚙️ Tech Stack  
- **Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn  
- **Deployment:** Streamlit  

---

## ▶️ Running the Project  

### 1. Clone the Repository  
```bash
git clone https://github.com/USERNAME/alzheimers-classification.git
cd alzheimers-classification
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Train the Model (if needed)  
```bash
jupyter notebook "Alzheimer's.ipynb"
```

### 4. Run the Streamlit App  
```bash
streamlit run app.ipynb
```

---

## 📊 Results  
- Achieved baseline classification performance with CNN.  
- Successfully deployed the model on Streamlit.  
  

---

📌 **Future Work**  
- Add **Grad-CAM** explainability for visual interpretation.  
- Improve accuracy using pre-trained CNN architectures (e.g., Xception, ResNet).  
- Deploy on cloud platforms for wider accessibility.  
