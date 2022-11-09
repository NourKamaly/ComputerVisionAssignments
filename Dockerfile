FROM python:3.9.14-slim

RUN pip install --no-cache-dir numpy==1.23.2 matplotlib==3.5.3
RUN pip install --no-cache-dir opencv-contrib-python
RUN pip install --no-cache-dir mlflow==1.28.0

COPY "1.NonMaximumSupression"  ./non_maximum_supression/

ENTRYPOINT ["tail","-f","/dev/null"]