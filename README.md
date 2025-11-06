# Replika: Semantic Code Clone Detection System

## 🧠 Overview
This project implements a **code clone detection system** supporting two major datasets:

- **BigCloneBench (BCB)**
- **POJ-104**

It leverages **SentenceTransformers** to generate semantic code embeddings and **Qdrant** vector database for efficient similarity search.

---

## ⚙️ Prerequisites
- **Python 3.12**
- **Docker** and **Docker Compose**

---

## 🚀 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/replika.git
cd replika
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧩 Setup

### Start Qdrant vector database
```bash
docker-compose up -d
```

### Prepare your datasets

#### For **BigCloneBench (BCB)**:
Place the dataset files in:
```
datasets/BigCloneBench/
```

**Required files:**
- `CLONES.csv`
- `FUNCTIONS_CLEANED.csv`

#### For **POJ-104**:
Place the dataset in:
```
datasets/poj104/
```

---

## 💻 Usage
Run the system with command-line options:

```bash
python src/main.py [OPTIONS]
```

### Command Line Options
| Option | Description | Default |
|---------|--------------|----------|
| `--dataset` | Choose dataset (`bcb` or `poj`) | `bcb` |
| `--qdrant_host_url` | Qdrant host URL | `localhost` |
| `--qdrant_port` | Qdrant port | `6333` |
| `--normalized` | Whether to normalize embeddings | `False` |
| `--embedding_model` | SentenceTransformer model name | `"all-MiniLM-L6-v2"` |
| `--num_samples` | Number of samples for benchmarking | `500` |
| `--k` | Number of nearest neighbors to retrieve | `100` |

---

## 🔧 Examples

### Run with **BCB** dataset:
```bash
python src/main.py --dataset bcb --num_samples 1000 --k 100
```

### Run with **POJ-104** dataset:
```bash
python src/main.py --dataset poj --embedding_model "microsoft/codebert-base"
```

---

## 📊 Benchmarking
The system performs automatic benchmarking using the following metrics:

- **Success Rate at k**
- **Mean Precision at k**
- **Mean Reciprocal Rank (MRR)**
- **Mean Average Precision at k (MAP@k)**

---

## 📁 Project Structure
```
replika/
├── src/
│   ├── main.py              # Main entry point
│   ├── bcb_utils.py         # BigCloneBench utilities
│   ├── poj_utils.py         # POJ-104 utilities
│   └── utils/
│       ├── logging.py       # Logging configuration
│       └── qdrant_utils.py  # Qdrant interface utilities
├── datasets/                # Dataset directory
├── database/                # Qdrant storage
├── docker-compose.yml       # Docker configuration
└── requirements.txt         # Python dependencies
```





