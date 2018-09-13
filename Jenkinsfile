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
            def build_image = docker.build("build_env:${env.BUILD_ID}")

            build_image.withRun('-p 3307:3307') {
                /* do things */
                sh '''echo "Hello there"'''
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
                        ./saxpy/build/saxpy
                    '''
                }
            }
        }
    }
}
