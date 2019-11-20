FROM floydhub/tensorflow:1.14-py3_aws.44 

RUN mkdir /app
RUN cd /app && git clone https://github.com/sadeepj/crfasrnn_keras.git
RUN cd /app/crfasrnn_keras/src/cpp && make
ENV PYTHONPATH=/app/crfasrnn_keras/src
RUN wget --directory-prefix=/app/crfasrnn_keras/model https://github.com/sadeepj/crfasrnn_keras/releases/download/v1.0/crfrnn_keras_model.h5

