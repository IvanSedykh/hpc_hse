export USER_AT_HOST="idsedykh@cluster.hpc.hse.ru"
export PUBKEYPATH="$HOME/.ssh/id_rsa.pub"

ssh-copy-id -i "$PUBKEYPATH" "$USER_AT_HOST"