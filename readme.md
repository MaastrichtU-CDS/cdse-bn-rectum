#Commands
- Pull new changes in client: git submodule foreach git pull origin master
- Build container: docker build . -t lery/bn-test:latest
- Push to remote registry: docker push lery/bn-test:latest