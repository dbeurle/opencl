pipeline {
    agent any
        stage('ocl-example') {

            docker.build("build_env:${env.BUILD_ID}").withRun('--device=/dev/dri:/dev/dri') {
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
