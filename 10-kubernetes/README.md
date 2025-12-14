Kubernetes a.k.a K8s is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications.

Docker is used to create an image of your running application. While Docker run can run a container from an image, it doesn't provide any way to manage the container's lifecycle or scale the application which is why Kubernetes is used.

Kubernetes is a container orchestration platform that provides a way to manage the lifecycle of containers and scale applications.
In k8s, the running container is called a pod. 

The pod is the smallest unit of deployment in k8s. We can increase/decrease the number of pods to scale up/down the application as per the load. This feature in k8s is called Horizontal Pod Autoscaling(HPA).

Entrypoint to your pod(app runs on docker image): Ingress -> External Services(Gateway service)->Internal Services(Pods)
Ingress is like a traffic cop for your application. Ingress routes the traffic to the appropriate service based on the URL.