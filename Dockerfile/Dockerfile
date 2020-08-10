FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter
#
###
ENV DEBIAN_FRONTEND=noninteractive
#RUN sed -i 's/archive.ubuntu.com/free.nchc.org.tw/g' /etc/apt/sources.list
RUN apt update && apt install wget -y && apt install vim -y
#RUN echo deb http://deb.obspy.org xenial main >> /etc/apt/sources.list
#RUN wget --quiet -O - https://raw.github.com/obspy/obspy/master/misc/debian/public.key | apt-key add -
RUN apt update
#RUN apt install -y openssh-server python3-obspy -y
RUN apt install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
###
RUN pip install --upgrade pip
RUN pip install keras==2.2.0
##
EXPOSE 8888
EXPOSE 22
###
COPY .keras /root/.keras
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
#COPY tensorflow-for-poets-2 /home/tensorflow-for-poets-2
COPY CNN_Mnist.ipynb /home/CNN_Mnist.ipynb
COPY RNN_Mnist.ipynb /home/RNN_Mnist.ipynb
COPY CNN_Retrain.ipynb /home/CNN_Retrain.ipynb
COPY flower_test4.jpeg /home/flower_test4.jpeg
COPY flower_test2.jpeg /home/flower_test2.jpeg
COPY run_jupyter.sh run_jupyter.sh
COPY django-upload-example django-upload-example
RUN pip install -r django-upload-example/requirements.txt
USER root
COPY run_jupyter.sh run_jupyter.sh
CMD ./run_jupyter.sh
#CMD ["source /etc/bash.bashrc && jupyter notebook --notebook-dir=/home --ip 0.0.0.0 --no-browser --allow-root"]
