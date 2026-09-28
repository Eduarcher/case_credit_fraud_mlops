# Use an official Python runtime as a parent image
FROM public.ecr.aws/docker/library/python:3.11.9-slim as dependencies

# Setup environment
ENV APP_HOME "/app"
WORKDIR ${APP_HOME}
USER root

# Add files
ADD credit_fraud ${APP_HOME}/credit_fraud
ADD dependencies ${APP_HOME}/dependencies
ADD models ${APP_HOME}/models
COPY config.yml ${APP_HOME}
COPY pyproject.toml ${APP_HOME}
COPY setup.py ${APP_HOME}

# Install project
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org . 

CMD ["cf-run"] 
USER 1001