#FROM yusongwang1991/mnist:latest
FROM yusongwang1991/mnist:v1
RUN rm -rf django-upload-example
COPY django-upload-example django-upload-example 
COPY run_jupyter.sh run_jupyter.sh
COPY model model
#COPY number.py number.py
CMD ./run_jupyter.sh
#CMD ["source /etc/bash.bashrc && jupyter notebook --notebook-dir=/home --ip 0.0.0.0 --no-browser --allow-root"]
