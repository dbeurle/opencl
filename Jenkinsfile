pipeline {
    agent any
    stages {
        stage('ocl-example') {
            agent {
                dockerfile {
                    filename 'Dockerfile'
                    additionalBuildArgs '--pull'
                }
            }
            steps {
                sh '''
                cd saxpy
                '''
            }
        }
    }
}
