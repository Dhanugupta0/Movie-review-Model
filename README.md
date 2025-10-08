🎬 IMDB Movie Review Sentiment Analysis

A deep learning project that classifies IMDB movie reviews as positive or negative using a Simple RNN model. The project includes model training, sentiment prediction, and an interactive Streamlit web application for real-time analysis.

🚀 Features

🔹 Binary Sentiment Classification – Predicts reviews as positive or negative

🔹 Simple RNN Architecture – Lightweight recurrent neural network with an embedding layer

🔹 IMDB Dataset – Trained on 25,000 movie reviews with top 10,000 frequent words

🔹 Text Preprocessing – Tokenization, encoding, and padding pipeline

🔹 Interactive Web App – Streamlit-based app for real-time predictions

🔹 Confidence Scores – Provides probability values with sentiment labels

🧠 Model Architecture
Layer	Output Shape	Parameters
Embedding	(32, 500, 128)	1,280,000
SimpleRNN	(32, 128)	32,896
Dense (Sigmoid)	(32, 1)	129
Total	—	1,313,027 (~5.01 MB)

Configuration

Vocabulary size: 10,000 words

Sequence length: 500 tokens

Embedding dimension: 128

RNN units: 128

Batch size: 32

Epochs: 10 (with early stopping)

📊 Performance

✅ Training Accuracy: 90.54%

✅ Validation Accuracy: 74.58%

✅ Training Loss: 0.2504

✅ Validation Loss: 0.6678


Best Validation Performance (Epoch 5):

Accuracy: 79.62%

Loss: 0.5115

⚙️ Installation
# Clone the repository
git clone https://github.com/Dhanugupta0/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Install dependencies
pip install numpy tensorflow streamlit

🖥️ Usage
🔹 Train the Model

Run the Jupyter notebook to train from scratch:

jupyter notebook main.ipynb


➡️ Model will be saved as simpleRnn_imdb.h5

🔹 Make Predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from utils import predict_sentiment   # (if you create a helper function)

# Load model
model = load_model('simpleRnn_imdb.h5')

# Sample Output
```bash
Review: This movie was fantastic! The acting was great and the plot was thrilling
Sentiment: Positive
Prediction Score: 0.7206853628158569
```

🔹 Run the Web App
streamlit run main.py


➡️ Opens an interactive interface to type reviews and get real-time predictions.

```bash
📂 Project Structure
├── main.ipynb          # Model training notebook
├── prediction.ipynb    # Testing and prediction notebook
├── main.py             # Streamlit app
├── simpleRnn_imdb.h5   # Trained model
└── README.md           # Documentation
```

🔧 Limitations

⚠️ The model shows overfitting – high training accuracy but lower validation performance.

Possible improvements:

Add Dropout for regularization

Use LSTM/GRU instead of SimpleRNN

Integrate Attention Mechanisms

Increase embedding dimensions

Apply data augmentation

🔮 Future Enhancements

✅ Replace RNN with LSTM/GRU

✅ Implement Attention-based models

✅ Extend to multi-class sentiment analysis

✅ Deploy as a REST API

✅ Visualize attention weights

✅ Batch predictions support

📜 License

This project is licensed under the MIT License.

🙌 Acknowledgments

IMDB dataset from TensorFlow/Keras

Built with TensorFlow and Streamlit
