# Document Title

## Jupyter server



## Steps on EC2:
### Configure the Jupyter server
$ jupyter notebook password

$ cd ~
$ mkdir ssl
$ cd ssl
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem

### Start the Jupyter notebook server
jupyter notebook --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key


## Configure Linux/Mac client

For ubuntu:
ssh -i ~/mykeypair.pem -N -f -L 8888:localhost:8888 ubuntu@ec2-###-##-##-###.compute-1.amazonaws.com

For Amazon linux:
ssh -i ~/mykeypair.pem -N -f -L 8888:localhost:8888 ubuntu@ec2-###-##-##-###.compute-1.amazonaws.com
