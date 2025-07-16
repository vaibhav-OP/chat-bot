
# **Neural Network Chatbot**
A PyTorch-based chatbot trained on custom intents to handle customer interactions and FAQs. The system implements a feedforward neural network with bag-of-words preprocessing for natural language understanding.

## Technical Implementation

**Core Components:**

**Data Processing:**
 - Tokenization and lemmatization of input patterns
 - Bag-of-words vectorization
 - Intent classification label preparation

**Neural Network Architecture:**

    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
 - 3-layer feedforward network with ReLU activation
 - Configurable input/hidden/output dimensions

**Training Pipeline:**

 - Cross-entropy loss and Adam optimizer
 - GPU/CPU compatible training loop
 - Model checkpointing

**Key Specifications:**
- Input size: Vocabulary length (bag-of-words)
- Hidden size: 16 neurons (configurable)
- Output size: Number of intent tags
- Learning rate: 0.0005
- Training epochs: 2000

## Usage

1.  Prepare your intents in  `intents.json`
2. Run the `chat-bot.ipynb` 
