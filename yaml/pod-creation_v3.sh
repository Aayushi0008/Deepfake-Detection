echo "Enter container image name"
read container

name=$(whoami)

cat <<EOF >$HOME/${name}-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: $name
spec:
  securityContext:
    runAsUser: $(id -u)
    fsGroup: $(id -g)
  containers:
  - name: $(whoami)
    image: $container
    env:
    - name: HOME
      value: /home/$(whoami)
    - name: NVIDIA_VISIBLE_DEVICES 
      value: none
    command: ["/bin/bash", "-c", "--" ]
    args: [ "while true; do sleep 1; done;" ]
    volumeMounts:
    - name: data
      mountPath: /workspace/data
    - name: nfs
      mountPath: /workspace/storage
    - name: home
      mountPath: /home
    - name: shadow
      mountPath: "/etc/shadow"
      readOnly: true
    - name: password
      mountPath: "/etc/passwd"
      readOnly: true
    - name: group
      mountPath: "/etc/group"
      readOnly: true
  volumes:
  - name: data
    hostPath:
      path: /raid/$(whoami)
      type: Directory
  - name: nfs
    hostPath:
      path: /DATA1/$(whoami)
      type: Directory
  - name: home
    hostPath:
      path: /home/$(whoami)
      type: Directory
  - name: group
    hostPath:
       path: "/etc/group"
  - name: shadow
    hostPath:
       path: "/etc/shadow"
  - name: password
    hostPath:
       path: "/etc/passwd"
  hostNetwork: true
  dnsPolicy: Default
EOF

cd $HOME
kubectl create -f ${name}-pod.yaml 
rm ${name}-pod.yaml
