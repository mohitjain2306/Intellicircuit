# IntelliCircuit ⚡
### AI-Powered Electrical Circuit Analysis & Optimization System

A web application that lets you describe electrical circuits in plain English and automatically performs engineering analysis, optimization, and visualization.

> **Live Demo:** https://intellicircuit.onrender.com

---

## What It Does

Type something like:
```
Calculate the output voltage of a voltage divider with R1=10k, R2=2.2k and 12V input
```
And the system will:
- Parse your natural language input
- Detect the circuit type automatically
- Perform all electrical calculations
- Optimize component values using a Genetic Algorithm
- Generate circuit diagrams and plots

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Optimization | Genetic Algorithm (custom built) |
| Diagrams | SVG (auto-generated) |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Render |

---

## Project Structure
```
Intellicircuit/
├── .github/
│   └── workflows/
│       └── deploy.yml         
├── backend/
│   ├── app.py                 
│   ├── requirements.txt       
│   ├── Procfile               
│   └── Dockerfile             
├── frontend/
│   ├── circuit_analyzer.html  
│   └── nginx.conf             
├── scripts/
│   └── deploy.sh              
├── docker-compose.yml         
├── .gitignore
└── README.md
```

---

## Run Locally

### Without Docker
```bash
git clone https://github.com/mohitjain2306/Intellicircuit.git
cd Intellicircuit
pip install -r backend/requirements.txt
cd backend
python app.py
```
Then open http://localhost:5000

### With Docker
```bash
git clone https://github.com/mohitjain2306/Intellicircuit.git
cd Intellicircuit
docker compose up
```
- Frontend: http://localhost:8080
- Backend: http://localhost:5001

---

## DevOps Setup

### Docker
The backend is containerized using Docker. The frontend is served using Nginx.
```bash
docker build -t intellicircuit-backend ./backend
docker run -p 5000:5000 intellicircuit-backend
```

### Docker Compose
Runs both frontend and backend together with one command:
```bash
docker compose up
```

### CI/CD Pipeline
Every push to main automatically:
1. Builds the Docker image
2. Pushes it to DockerHub

DockerHub image: `mohitjain2306/intellicircuit-backend`

---

## Example Queries
```
Calculate the output voltage of a voltage divider with R1=10k ohm, R2=2.2k ohm, and input voltage of 12V

Design an RC low-pass filter with R=1k ohm and C=100nF. Find the cutoff frequency.

Calculate the gain of a non-inverting op-amp with Rf=22k ohm and R1=2.2k ohm.

Analyze a series RLC circuit with R=100 ohm, L=10mH, and C=220nF.
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Check backend status |
| POST | `/api/circuit/analyze` | Analyze a circuit |
| POST | `/api/circuit/optimize` | Optimize component values |

---

## Circuit Types Supported

- Voltage Divider
- RC Filter (Low Pass / High Pass)
- RL Circuit
- RLC Circuit
- Op-Amp (Inverting / Non-Inverting)
- Series Resistor Network
- Parallel Resistor Network

---

## Author

**Mohit Jain**
- GitHub: [@mohitjain2306](https://github.com/mohitjain2306)
- Live App: [intellicircuit.onrender.com](https://intellicircuit.onrender.com)
- DockerHub: [mohitjain2306](https://hub.docker.com/u/mohitjain2306)