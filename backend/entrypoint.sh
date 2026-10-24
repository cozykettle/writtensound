#!/bin/sh
# Drop from root to a PUID/PGID user, grant access to the mounted DRI nodes,
# then exec the requested command. Unraid-style IDs are the defaults.
set -eu

PUID="${PUID:-99}"
PGID="${PGID:-100}"
WS_PORT="${WS_PORT:-8080}"

case "$PUID" in
    ''|*[!0-9]*) echo "entrypoint: PUID must be numeric" >&2; exit 1 ;;
esac
case "$PGID" in
    ''|*[!0-9]*) echo "entrypoint: PGID must be numeric" >&2; exit 1 ;;
esac

if [ "$PUID" -eq 0 ]; then
    APP_USER="root"
else
    if ! getent group "$PGID" >/dev/null 2>&1; then
        groupadd --gid "$PGID" writtensound
    fi
    GROUP_NAME="$(getent group "$PGID" | cut -d: -f1)"

    if getent passwd "$PUID" >/dev/null 2>&1; then
        APP_USER="$(getent passwd "$PUID" | cut -d: -f1)"
        usermod --gid "$PGID" "$APP_USER"
    else
        useradd --uid "$PUID" --gid "$PGID" --no-create-home \
            --shell /usr/sbin/nologin writtensound
        APP_USER="writtensound"
    fi

    for device in /dev/dri/card* /dev/dri/renderD*; do
        [ -e "$device" ] || continue
        device_gid="$(stat -c '%g' "$device")"
        if ! getent group "$device_gid" >/dev/null 2>&1; then
            groupadd --gid "$device_gid" "dri-$device_gid"
        fi
        device_group="$(getent group "$device_gid" | cut -d: -f1)"
        usermod --append --groups "$device_group" "$APP_USER"
    done
fi

umask 002
mkdir -p /data/config /data/jobs /data/models
chown "$PUID:$PGID" /data /data/config /data/jobs /data/models

if [ "$#" -eq 0 ]; then
    set -- uvicorn backend.app:create_app --factory \
        --host 0.0.0.0 --port "$WS_PORT"
fi

exec gosu "$APP_USER" "$@"
