# Replika: Code Clone Detection System

## 🧩 Overview
This project implements a **code clone detection system** that supports two major datasets:

- **BigCloneBench (BCB)**
- **POJ-104**

The system leverages **SentenceTransformers** to generate semantic code embeddings and uses the **Qdrant vector database** for efficient similarity search and retrieval.

---

## ⚙️ Prerequisites
Before running the project, make sure you have the following installed:
- **Python 3.12**
- **Docker** and **Docker Compose**

---

## 🚀 Installation

### 1. Clone the repository
git clone https://github.com/yourusername/replika.git
cd replika
### 2. Create and activate a virtual environment
bash
Copier le code
python -m venv venv
source venv/bin/activate      # On Windows: .\venv\Scripts\activate
3. Install the required dependencies
bash
Copier le code
pip install -r requirements.txt
⚡ Setup
#1. Start the Qdrant vector database
Use Docker Compose to launch Qdrant:

bash
Copier le code
docker-compose up -d
2. Prepare the datasets
For BigCloneBench (BCB):
Place the required dataset files in the following path:

swift
Copier le code
datasets/BigCloneBench/
Required files:

objectivec
Copier le code
CLONES.csv
FUNCTIONS_CLEANED.csv
For POJ-104:
Place the dataset files in:

bash
Copier le code
datasets/poj104/
💻 Usage
Run the system using the command line with various configuration options:

bash
Copier le code
python src/main.py [OPTIONS]
Command Line Options
Option	Description	Default
--dataset	Choose dataset (bcb or poj)	bcb
--qdrant_host_url	Qdrant host URL	localhost
--qdrant_port	Qdrant port	6333
--normalized	Whether to normalize embeddings	False
--embedding_model	SentenceTransformer model name	all-MiniLM-L6-v2
--num_samples	Number of samples for benchmarking	500
--k	Number of nearest neighbors to retrieve	100

Example Commands
Run with BigCloneBench:

bash
Copier le code
python src/main.py --dataset bcb --num_samples 1000 --k 100
Run with POJ-104:

bash
Copier le code
python src/main.py --dataset poj --embedding_model "microsoft/codebert-base"
📊 Benchmarking
The system automatically performs benchmarking using the following evaluation metrics:

Success Rate at k

Mean Precision at k

Mean Reciprocal Rank (MRR)

Mean Average Precision at k (MAP@k)

📁 Project Structure
"""bash
Copier le code
replika/
├── src/
│   ├── main.py               # Main entry point
│   ├── bcb_utils.py          # BigCloneBench utilities
│   ├── poj_utils.py          # POJ-104 utilities
│   └── utils/
│       ├── logging.py        # Logging configuration
│       └── qdrant_utils.py   # Qdrant interface utilities
├── datasets/                 # Dataset directory
├── database/                 # Qdrant storage
├── docker-compose.yml        # Docker configuration
└── requirements.txt          # Python dependencies


