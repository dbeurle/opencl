pipeline {
    agent any
    stages{
        stage('ocl-example') {
            agent {
                 dockerfile {
                     filename 'Dockerfile'
                     additionalBuildArgs ''
                     args 'asdf --device=/dev/dri:/dev/dri'
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
                    cd saxpy && mkdir build && cd build
                    cmake .. && make all
                    ./saxpy
                    '''
                }
            }
        }
    }
}
