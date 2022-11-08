FROM python:3.10.8-slim-bullseye

RUN pip install --no-cache-dir numpy pandas matplotlib 
RUN pip install --no-cache-dir opencv-contrib-python
RUN pip install --no-cache-dir mlflow

COPY "1.NonMaximumSupression"  ./non_maximum_supression/

ENTRYPOINT ["tail","-f","/dev/null"]