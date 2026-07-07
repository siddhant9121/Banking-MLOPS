pipeline {
    agent any

    environment {
        PROJECT_NAME   = 'banking-document-mlops'
        PYTHON_VERSION = '3.9'
        DOCKER_IMAGE   = "banking-doc-mlops:${BUILD_NUMBER}"
        VENV_DIR       = 'venv'
    }

    stages {

        stage('Checkout') {
            steps {
                echo '=== Checking out source code ==='
                checkout scm
                sh 'git log -1 --oneline'
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo '=== Creating virtual environment ==='
                sh """
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                """
            }
        }

        stage('Code Quality') {
            parallel {
                stage('flake8') {
                    steps {
                        sh """
                            . ${VENV_DIR}/bin/activate
                            flake8 src/ tests/ --max-line-length=100 --count --statistics || true
                        """
                    }
                }
                stage('black') {
                    steps {
                        sh """
                            . ${VENV_DIR}/bin/activate
                            black --check src/ tests/ || true
                        """
                    }
                }
                stage('isort') {
                    steps {
                        sh """
                            . ${VENV_DIR}/bin/activate
                            isort --check-only src/ tests/ || true
                        """
                    }
                }
            }
        }

        stage('Run Tests') {
            steps {
                echo '=== Running pytest suite ==='
                sh """
                    . ${VENV_DIR}/bin/activate
                    pytest tests/ -v \
                        --cov=src \
                        --cov-report=xml:coverage.xml \
                        --cov-report=html:htmlcov \
                        --junitxml=test-results.xml || true
                """
            }
            post {
                always {
                    junit 'test-results.xml'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }

        stage('Validate Config & Params') {
            steps {
                echo '=== Validating YAML configs ==='
                sh """
                    . ${VENV_DIR}/bin/activate
                    python3 -c "
import yaml, sys
with open('configs/model_config.yaml') as f:
    cfg = yaml.safe_load(f)
required = ['classifier', 'ner', 'data', 'training', 'validation']
missing = [k for k in required if k not in cfg]
if missing:
    print(f'model_config.yaml missing: {missing}')
    sys.exit(1)
print('model_config.yaml OK')
with open('params.yaml') as f:
    p = yaml.safe_load(f)
required_p = ['train', 'ner', 'evaluate']
missing_p = [k for k in required_p if k not in p]
if missing_p:
    print(f'params.yaml missing: {missing_p}')
    sys.exit(1)
print('params.yaml OK')
                    "
                """
            }
        }

        stage('Train Models') {
            steps {
                echo '=== Running training pipeline (dry-run) ==='
                sh """
                    . ${VENV_DIR}/bin/activate
                    python src/train.py --config configs/model_config.yaml
                """
            }
            post {
                success {
                    archiveArtifacts artifacts: 'models/**/*.pth', allowEmptyArchive: true
                    archiveArtifacts artifacts: 'mlruns/**',       allowEmptyArchive: true
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "=== Building Docker image: ${DOCKER_IMAGE} ==="
                sh "docker build -t ${DOCKER_IMAGE} ."
            }
        }

        stage('Test Docker Image') {
            steps {
                echo '=== Smoke-testing Docker container ==='
                sh """
                    docker run -d --name test-container-${BUILD_NUMBER} \
                        -p 8001:8000 ${DOCKER_IMAGE}
                    sleep 15
                    curl --retry 5 --retry-delay 3 --fail http://localhost:8001/health
                    docker stop  test-container-${BUILD_NUMBER}
                    docker rm    test-container-${BUILD_NUMBER}
                """
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo '=== Awaiting deployment approval ==='
                input message: 'Deploy banking-doc-mlops to production?', ok: 'Deploy'
                echo "=== Deploying ${DOCKER_IMAGE} to production ==="
                sh """
                    docker tag ${DOCKER_IMAGE} banking-doc-mlops:stable
                    echo "Deployment of ${DOCKER_IMAGE} completed successfully!"
                """
            }
        }
    }

    post {
        success {
            echo "Pipeline PASSED — build ${BUILD_NUMBER}"
            emailext(
                subject: "SUCCESS: ${JOB_NAME} #${BUILD_NUMBER}",
                body:    "Pipeline completed successfully. See ${BUILD_URL} for details.",
                to:      "mosambiswas999@gmail.com"
            )
        }
        failure {
            echo "Pipeline FAILED — build ${BUILD_NUMBER}"
            emailext(
                subject: "FAILED: ${JOB_NAME} #${BUILD_NUMBER}",
                body:    "Pipeline failed. See ${BUILD_URL} for details.",
                to:      "mosambiswas999@gmail.com"
            )
        }
        always {
            echo '=== Cleaning workspace ==='
            cleanWs()
        }
    }
}
