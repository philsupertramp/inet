#!/bin/bash

# Requires env vars:
# - USERNAME: username
# - PASSWORD: password
# - VPN_PASSWORD: the password for your VPN account
# - VPN_USR: username of VPN account
# - SERVER_ADDR: NAS server address
#
# to mount remote directories /home and /KInsektDaten from 141.64.103.52 into ./mnt
#


set -e
# VPN AUTH
VPN_ADDR=SOME_VPN:443
VPN_USR=${VPN_USR:-SOME_USER}
VPN_PASSWORD=${VPN_PASSWD}

# NAS AUTH
SERVER_ADDR=${SERVER_ADDR}

password="${PASSWORD}"
username="${USERNAME}"

if [[ $UID != 0 ]]; then
    echo "Please run this script with sudo:"
    echo "sudo $0 $*"
    exit 1
fi

mkdir -p mnt/KInsektDaten
mkdir -p mnt/home


connect-vpn () {
  sudo openfortivpn ${VPN_ADDR} --username=${VPN_USR} --password=${VPN_PASSWORD} > openfortivpn.log 2>&1 &
}

disconnect-vpn () {
  pgrep -f "sudo openfortivpn" | xargs kill

}

mount-nas-dir() {
  sudo mount -t cifs -v -ousername=${username},password="${password}" //${SERVER_ADDR}/$1 mnt/$1
}


case $1 in
  vpn)
    case $2 in
      on)
        connect-vpn
        ;;
      off)
        disconnect-vpn
        ;;
    esac
    ;;
  on)
    connect-vpn
    sleep 5
    mount-nas-dir KInsektDaten
    mount-nas-dir home
    ;;
  off)
    sudo umount mnt/KInsektDaten
    sudo umount mnt/home
    disconnect-vpn
    ;;
  *)
    echo -e "Usage: ./mount_directories.sh [on|off]\n" \
         "\ton: Establish VPN connection and mount volumes\n" \
         "\toff: Unmount volumes and disconnect from VPN\n" \
         "\tvpn [on|off]: only (dis)connect VPN";
    ;;
esac
