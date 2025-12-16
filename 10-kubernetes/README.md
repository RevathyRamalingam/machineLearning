## Kubernetes a.k.a K8s is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications.

## Docker is used to create an image of your running application. While Docker run can run a container from an image, it doesn't provide any way to manage the container's lifecycle or scale the application which is why Kubernetes is used.

## Kubernetes is a container orchestration platform that provides a way to manage the lifecycle of containers and scale applications. In k8s, the running container is called a pod. 

## The pod is the smallest unit of deployment in k8s. We can increase/decrease the number of pods to scale up/down the application as per the load. This feature in k8s is called Horizontal Pod Autoscaling(HPA).

## Entrypoint to your pod(app runs on docker image): Ingress -> External Services(Gateway service)->Internal Services(Pods)


## Ingress is like a traffic cop for your application. Ingress routes the traffic to the appropriate External service(LoadBalancer) based on the URL.

## Ingress- Entry point to cluster/application
## Internal service - IP Cluster

## EKS Elastic Kubernetes Service is a managed k8s service provided by AWS without the need to install Kubernetes and manage our own kubernetes control plane.

A standalone Docker container cannot be directly detected by Kubernetes. It must be part of a pod within a Kubernetes cluster, which can be created using tools like KIND (Kubernetes in Docker) for local clusters.

In Kubernetes world, resources are managed in a cluster and all the resources in the cluster are called kind.
Kind means Kubernetes in Docker.
A kind could be a pod, a deployment, a service, a configmap, a secret, etc.
Cluster is the basement on which all the resources are built and controlled in the control plane.

## Kubernetes Architecture
Kubernetes' architecture is unique, comprising a control plane that manages its nodes. To facilitate this, Kubernetes uses KIND (Kubernetes in Docker), a tool that creates clusters and helps manage the nodes within them.

## The First Step: Loading Docker Images into Kubernetes with KIND

 Kubernetes cannot identify the docker daemon running on your local machine. It can only identify the docker daemon running inside the cluster. So we need to put our nodes into the kubernetes control plane. Load image into kubernetes control plane. This is the first step in kubernetes.

kind load docker-image clothing-classifier:v1 --name mlzoomcamp

## KIND Resources

In Kubernetes, KIND can refer to various resources, such as Pods, Deployments, Services, ConfigMaps, Secrets, and more. These resources are loaded into the cluster via KIND to facilitate the deployment and management of applications.

## Pod: The Smallest Unit of Deployment

A Pod is the smallest deployable unit in Kubernetes, essentially a containerized application that runs within the cluster.

## Deployment: Metadata for Pods

A Deployment acts as metadata that describes how Pods should be run. It defines the Pod's name, how much memory pod needs, its CPU utilization, etc.

## Ingress controller acts as traffic cop and diverts the traffic HTTP/S to the appropriate service.

## HPA - Horizontal Pod AutoScaler is a feature in k8s to scale pods up/down as per the traffic. In hpa.yaml file we specify the kind, name and specs like min, max replicas and under what conditions(eg. CPU itilization > 50%) it must scale.
For HPA to work, we need to run the MetricsServer that can post metrics information like CPU utilization, memory usage, etc about the pod to K8s via Kubernetes API.
Kubernetes service node port should be between valid ports 30000-32767. So in service.yaml we specify nodeport in this range.


