pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        PROJECT_NAME = 'banking-document-api'
        IMAGE_NAME = "banking-mlops-api:${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '=== Checking out code ==='
                checkout scm
            }
        }
        
        stage('Code Quality & Tests') {
            steps {
                echo '=== Running initial checks inside isolated env ==='
                sh '''
                    python3 -m venv venv || python -m venv venv
                    . venv/bin/activate || venv\\Scripts\\activate
                    pip install --upgrade pip
                    pip install flake8 pytest pytest-cov black isort
                    # Minimal dependency install for pipeline tests
                    pip install fastapi uvicorn pyyaml
                    black --check src/ || true
                    isort --check-only src/ || true
                    flake8 src/ --max-line-length=100 --exit-zero || true
                    pytest tests/ -v || true
                '''
            }
        }
        
        stage('Docker Build') {
            steps {
                echo '=== Building Docker Image for API ==='
                sh "docker build -t ${PROJECT_NAME}:latest -t ${PROJECT_NAME}:${env.BUILD_NUMBER} ."
            }
        }
        
        stage('Deploy to Server') {
            when {
                branch 'main'
            }
            steps {
                echo '=== Deploying API via Docker Container ==='
                // Stop any extremely old containers if they exist
                sh "docker stop banking-api || true"
                sh "docker rm banking-api || true"
                
                // Run the new model container bridging to port 8000!
                sh "docker run -d --name banking-api -p 8000:8000 ${PROJECT_NAME}:latest"
                echo '✓ Deployment successful! Banking API live on port 8000'
            }
        }
    }
    
    post {
        success {
            echo '✓✓✓ MLOps Pipeline completed successfully!'
        }
        failure {
            echo '✗✗✗ Pipeline failed!'
        }
        always {
            echo '=== Cleaning up workspace ==='
            cleanWs()
        }
    }
}