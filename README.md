# Banking Document Pipeline

This repository contains the Machine Learning pipeline and API for processing banking documents, classifying document types, and extracting entities like PAN, Aadhaar, DOB, and other key details.

## Project Architecture

The architecture relies on the following tools:
- **FastAPI**: Serves the trained models.
- **Docker**: Containerizes the application and MLflow tracking server.
- **Docker Compose**: Orchestrates the multi-container setup.
- **MLflow**: Tracks model training metrics and parameters.
- **DVC (Data Version Control)**: Manages the machine learning pipeline (DAG) and data lineage.
- **Kubernetes**: Production deployment orchestration.
- **CI/CD**: Jenkins and GitHub Actions workflows for automated code quality checks, tests, and Docker builds.

## Setup Instructions

### 1. Python Environment

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. DVC Pipeline

Initialize and run the DVC pipeline (Data Version Control):

```bash
dvc init
dvc dag
dvc repro
```

This will run the stages defined in `dvc.yaml` (`train`, `evaluate`) tracking `params.yaml`.

### 3. Docker & MLflow Setup

Start the MLflow tracking server and FastAPI server locally via Docker Compose:

```bash
docker-compose up -d --build
```
- The FastAPI application will be available at `http://localhost:8000/docs`
- The MLflow UI will be available at `http://localhost:5000`

### 4. Kubernetes Setup

To deploy this application to a local or production Kubernetes cluster (e.g., Minikube):

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check the status of your pods:
```bash
kubectl get pods
kubectl get services
```

### 5. CI/CD pipelines
This project uses **GitHub Actions** (see `.github/workflows/ci.yml`) and **Jenkins** (see `Jenkinsfile`) to automate testing and docker building whenever new commits are pushed.

## Project Structure

- `configs/`: YAML configuration files for models and API.
- `src/`: Python source code for training, data processing, and API.
- `tests/`: Unit test scripts.
- `models/`: Directory to persist trained PyTorch/Transformers models.
- `mlruns/`: Stores MLflow run data.
- `k8s/`: Kubernetes manifest files for deployment and services.
- `dvc.yaml`: DVC pipeline file.
- `params.yaml`: Training parameters tracked by DVC.

## Author

**Parth-Thorat-27** and Contributors
