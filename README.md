# SONAR Rock vs Mine Prediction

## Project Overview
This project implements a machine learning model to classify underwater objects as either rocks or mines using SONAR data. The system uses Logistic Regression to analyze SONAR signatures and make accurate predictions, which is crucial for maritime safety and underwater threat detection.

## 🚀 Features
- Processes SONAR signal data with 60 different attributes
- Uses Logistic Regression for binary classification
- Achieves high accuracy in distinguishing between rocks and mines
- Includes data preprocessing and model evaluation
- Provides real-time prediction capability for new SONAR signatures

## 🛠️ Technologies Used
- Python
- scikit-learn
- numpy
- pandas

## 📊 Dataset
The project uses the SONAR dataset containing patterns obtained from various angles and under different conditions. Each pattern is a set of 60 numbers in the range 0.0 to 1.0, representing the energy within a particular frequency band over a certain period.

## 💡 Model Performance
- Training Data Accuracy: ~83%
- Testing Data Accuracy: ~84%
- The model shows consistent performance across both training and test sets, indicating good generalization

## 🔧 Setup and Installation
1. Clone the repository
```bash
git clone [your-repository-link]
```

2. Install required dependencies
```bash
pip install numpy pandas scikit-learn
```

3. Run the application
```bash
python app.py
```

## 📈 Usage
The system can process new SONAR signatures in the following format:
```python
input_data = [60 numerical values between 0.0 and 1.0]
```
The model will then classify the object as either a rock or a mine.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests. You can also open issues for any bugs found or improvements suggested.

## 📝 License
This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact
Krishna - [LinkedIn](https://www.linkedin.com/in/krishnatanwars/)

Project Link: [GitHub](https://github.com/KrishnaTanwar/SONAR_Rock_vs_Mine_Prediction)