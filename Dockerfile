FROM floydhub/tensorflow:1.14-py3_aws.44 

RUN mkdir /app
COPY . /app/ 
RUN wget --directory-prefix=/app/model https://github.com/sadeepj/crfasrnn_keras/releases/download/v1.0/crfrnn_keras_model.h5
