TensorFlow being a large framework needs several GBs to be deployed in AWS Lambda which is a serverless computing available by AWS. 
A heavy model framework would increase the size of the container image, increasing time to load the model and thereby increases the cost to use the resources. So TensorFlow Lite was introduced to address this demand. TF Lite is part of TensorFlow package.

Serverless means you dont own any server to use it locally, instead hire the server rented by AWS and pay as you use.

Steps in TensorFlow Lite
* Model is converted to TFLite format which is lesser in size and hence improves the performance.
* TFlite format model is interpreted to allocate tensors
* Using interpreter we can set Tensor(input) and getTensor(predictions as output)

Amazon ECR Public Gallery - the place where amazon base images are available
Create a docker image of package 
Expose lambda function via API Gateway
It is possible to create a Lambda Function from scratch or by container image in AWS.

In pytorch, the model is saved in ONNX(Open Neural Network Exchange) format which can be used by webservices for predicting the outputs.
