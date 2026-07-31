#!/bin/bash
# Debug access only: if RunPod injected a PUBLIC_KEY (Pod deploys always get
# one), accept it and start sshd so a pod running this same image can be
# SSHed into. Harmless no-op on Serverless, which never sets PUBLIC_KEY.
set -e
if [ -n "$PUBLIC_KEY" ]; then
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd
fi
exec "$@"
