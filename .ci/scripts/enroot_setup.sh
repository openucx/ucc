#!/bin/bash -xe

# Configure enroot credentials if provided
if [[ -n "${ENROOT_USERNAME}" && -n "${ENROOT_PASSWORD}" ]]; then
    ENROOT_CONFIG_DIR="${HOME}/.config/enroot"
    ENROOT_CREDENTIALS_FILE="${ENROOT_CONFIG_DIR}/.credentials"
    mkdir -p "${ENROOT_CONFIG_DIR}"
    ENROOT_HOST="${ENROOT_REGISTRY}"
    ENROOT_CREDS_LINE="machine ${ENROOT_HOST} login ${ENROOT_USERNAME} password ${ENROOT_PASSWORD}"

    # If file exists, replace existing line for ENROOT_HOST, or append if not present
    if [[ -f "${ENROOT_CREDENTIALS_FILE}" ]]; then
        if grep -qE "^machine[[:space:]]+${ENROOT_HOST}[[:space:]]+login[[:space:]]+" "${ENROOT_CREDENTIALS_FILE}"; then
            sed -i "s|^machine[[:space:]]\+${ENROOT_HOST}[[:space:]]\+login[[:space:]]\+.*\$|${ENROOT_CREDS_LINE}|" "${ENROOT_CREDENTIALS_FILE}"
        else
            echo "${ENROOT_CREDS_LINE}" >> "${ENROOT_CREDENTIALS_FILE}"
        fi
    else
        echo "${ENROOT_CREDS_LINE}" > "${ENROOT_CREDENTIALS_FILE}"
    fi

    chmod 600 "${ENROOT_CREDENTIALS_FILE}"
fi
