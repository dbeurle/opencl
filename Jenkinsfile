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
                cd saxpy && mkdir build && cd build
                cmake .. && make all
                '''
            }
            post {
                success {
                    sh '''
                        cd build
                        ./saxpy
                    '''
                }
            }
        }
    }
}
