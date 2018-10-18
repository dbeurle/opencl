pipeline {
    agent any
    stages{
        stage('ocl-example') {
            agent {
                 dockerfile {
                     filename 'Dockerfile'
                 }
            }
            steps {
                sh '''
                clinfo
                clpeak
                cd saxpy && mkdir build && cd build
                cmake .. && make all
                '''
            }
            post {
                success {
                    sh '''
                    ./saxpy/build/saxpy
                    '''
                }
            }
        }
    }
}
