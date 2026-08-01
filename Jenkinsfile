pipeline {
    agent any

    environment {
        IMAGE_NAME     = 'travel-planner'
        CONTAINER_NAME = 'travel-planner-app'
        APP_PORT       = '8501'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'ls -la'
            }
        }

        stage('Build image') {
            steps {
                sh 'docker build -t $IMAGE_NAME:$BUILD_NUMBER -t $IMAGE_NAME:latest .'
            }
        }

        stage('Deploy') {
            steps {
                withCredentials([
                    string(credentialsId: 'GROQ_API_KEY',           variable: 'GROQ_API_KEY'),
                    string(credentialsId: 'DATABASE_URL',           variable: 'DATABASE_URL'),
                    string(credentialsId: 'AVIATION_STACK_API_KEY', variable: 'AVIATION_STACK_API_KEY'),
                    string(credentialsId: 'OPENWEATHER_API_KEY',    variable: 'OPENWEATHER_API_KEY'),
                    string(credentialsId: 'TAVILY_API_KEY',         variable: 'TAVILY_API_KEY')
                ]) {
                    sh '''
                        docker rm -f $CONTAINER_NAME || true

                        docker run -d \
                          --name $CONTAINER_NAME \
                          --restart unless-stopped \
                          -p $APP_PORT:$APP_PORT \
                          -e GROQ_API_KEY="$GROQ_API_KEY" \
                          -e DATABASE_URL="$DATABASE_URL" \
                          -e AVIATION_STACK_API_KEY="$AVIATION_STACK_API_KEY" \
                          -e OPENWEATHER_API_KEY="$OPENWEATHER_API_KEY" \
                          -e TAVILY_API_KEY="$TAVILY_API_KEY" \
                          $IMAGE_NAME:latest
                    '''
                }
            }
        }

        stage('Verify') {
            steps {
                sh 'sleep 10 && docker ps --filter name=$CONTAINER_NAME'
            }
        }
    }

    post {
        success {
            echo "Deployed. App should be reachable on port $APP_PORT"
        }
        failure {
            sh 'docker logs $CONTAINER_NAME --tail 50 || true'
        }
    }
}