ECR_URL=359875384121.dkr.ecr.us-east-1.amazonaws.com
LOCAL_IMAGE=churn-prediction-lambda

REPO_URL =${ECR_URL}/${LOCAL_IMAGE}
docker build -t ${LOCAL_IMAGE} .
REMOTE_IMAGE_TAG="${ECR_URL}/${LOCAL_IMAGE}:v1"

#docker build -t churn-prediction-lambda .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}