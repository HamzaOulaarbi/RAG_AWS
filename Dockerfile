# FROM python:3.10
FROM public.ecr.aws/lambda/python:3.10
WORKDIR /app
COPY requirements.txt ./
EXPOSE 8080
# Copy all files in ./src
COPY src/ ./
COPY . ./

# COPY ./src ./
RUN pip install -r requirements.txt

# ENV STREAMLIT_SERVER_PORT 8081
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8080"]

# CMD ["myfunction.lambda_handler"]
#, "--server.address=0.0.0.0" ]


