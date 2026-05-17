#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${APP_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${APP_DIR}/.env"
  set +a
fi

: "${DOMAIN:?Set DOMAIN in app/.env (or export DOMAIN)}"
: "${LETSENCRYPT_EMAIL:?Set LETSENCRYPT_EMAIL in app/.env (or export LETSENCRYPT_EMAIL)}"

STAGING="${CERTBOT_STAGING:-0}"
if [ "${STAGING}" = "1" ]; then
  echo "[init-letsencrypt] Using Let's Encrypt staging environment"
  STAGING_ARG="--staging"
else
  STAGING_ARG=""
fi

CONF_DIR="${APP_DIR}/certbot/conf"
WEBROOT_DIR="${APP_DIR}/certbot/www"

mkdir -p "${CONF_DIR}" "${WEBROOT_DIR}"

if [ ! -d "${CONF_DIR}/live/${DOMAIN}" ]; then
  echo "[init-letsencrypt] Creating dummy certificate for ${DOMAIN}..."
  mkdir -p "${CONF_DIR}/live/${DOMAIN}"
  openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
    -keyout "${CONF_DIR}/live/${DOMAIN}/privkey.pem" \
    -out "${CONF_DIR}/live/${DOMAIN}/fullchain.pem" \
    -subj "/CN=${DOMAIN}" >/dev/null 2>&1
fi

echo "[init-letsencrypt] Starting nginx (and app)..."
cd "${APP_DIR}"
docker compose up -d nginx

echo "[init-letsencrypt] Requesting Let's Encrypt certificate for ${DOMAIN}..."
rm -rf "${CONF_DIR}/live/${DOMAIN}" "${CONF_DIR}/archive/${DOMAIN}" "${CONF_DIR}/renewal/${DOMAIN}.conf" || true

docker compose run --rm certbot certonly \
  --webroot -w /var/www/certbot \
  ${STAGING_ARG} \
  --email "${LETSENCRYPT_EMAIL}" \
  --agree-tos \
  --no-eff-email \
  -d "${DOMAIN}"

echo "[init-letsencrypt] Reloading nginx..."
docker compose exec nginx nginx -s reload

echo "[init-letsencrypt] Done. HTTPS should now be active on https://${DOMAIN}"
