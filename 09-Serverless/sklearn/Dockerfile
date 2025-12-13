FROM agrigorev/model-2025-hairstyle:v1
#COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
RUN pip install onnxruntime scikit-learn keras-image-helper

WORKDIR /var/task
COPY pyproject.toml uv.lock ./

#RUN uv sync - skikit-learn module not found error occurs when docker is run
#Inorder to avoid that we are installing the dependencies using pip install globally so that
# AWS Lambda can find them ,it is added above 
#RUN uv pip install --system -r <(uv export --format requirements-txt)

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]