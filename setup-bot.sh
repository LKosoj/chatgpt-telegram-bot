#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"
ENV_FILE="$SCRIPT_DIR/.env"
SERVICE_NAME="chatgpt-telegram-bot"
DEFAULT_SYSTEMD_SERVICE_NAME="tg_bot"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
SYSTEMD_UNIT_DIR="/etc/systemd/system"

COMPOSE_STYLE=""

copy_env_example() {
    if [ ! -f "$ENV_EXAMPLE" ]; then
        printf 'Missing %s\n' "$ENV_EXAMPLE" >&2
        exit 1
    fi

    if [ ! -f "$ENV_FILE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        printf 'Created %s from %s\n' "$ENV_FILE" "$ENV_EXAMPLE"
    else
        printf 'Using existing %s\n' "$ENV_FILE"
    fi
}

env_value() {
    key="$1"
    awk -v key="$key" '
        {
            line = $0
            if (line ~ /^[[:space:]]*#/) {
                next
            }
            sub(/^[[:space:]]*export[[:space:]]+/, "", line)
            if (line ~ "^[[:space:]]*" key "[[:space:]]*=") {
                sub(/^[^=]*=/, "", line)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
                print line
                exit
            }
        }
    ' "$ENV_FILE"
}

is_unfilled_value() {
    key="$1"
    value="$2"

    case "$value" in
        ""|"''"|\"\"|"XXX"|\"XXX\")
            return 0
            ;;
    esac

    case "$key:$value" in
        "OPENAI_API_KEY:your_openai_api_key"|\
        "OPENAI_API_KEY:\"your_openai_api_key\""|\
        "TELEGRAM_BOT_TOKEN:your_telegram_bot_token"|\
        "TELEGRAM_BOT_TOKEN:\"your_telegram_bot_token\""|\
        "OPENAI_BASE_URL:https://example.com/v1/"|\
        "OPENAI_BASE_URL:\"https://example.com/v1/\""|\
        "OPENAI_BASE_URL:http://gateway.example/v1"|\
        "OPENAI_BASE_URL:\"http://gateway.example/v1\""|\
        "OPENAI_BASE_URL:http://gateway.example/v1/"|\
        "OPENAI_BASE_URL:\"http://gateway.example/v1/\"")
            return 0
            ;;
    esac

    return 1
}

set_env_value() {
    key="$1"
    value="$2"
    tmp_file="${ENV_FILE}.tmp.$$"

    awk -v key="$key" -v value="$value" '
        BEGIN {
            done = 0
        }
        {
            line = $0
            test_line = line
            sub(/^[[:space:]]*export[[:space:]]+/, "", test_line)
            if (!done && test_line ~ "^[[:space:]]*" key "[[:space:]]*=") {
                print key "=" value
                done = 1
                next
            }
            print line
        }
        END {
            if (!done) {
                print key "=" value
            }
        }
    ' "$ENV_FILE" > "$tmp_file"

    mv "$tmp_file" "$ENV_FILE"
}

read_prompt_value() {
    secret="$1"
    value=""

    if [ "$secret" = "true" ] && [ -t 0 ]; then
        stty_state="$(stty -g)"
        stty -echo
        IFS= read -r value || value=""
        stty "$stty_state"
        printf '\n'
    else
        IFS= read -r value || value=""
    fi

    printf '%s' "$value"
}

prompt_env_value() {
    key="$1"
    label="$2"
    secret="$3"
    current_value="$(env_value "$key")"

    if ! is_unfilled_value "$key" "$current_value"; then
        printf '%s is already set; skipping prompt.\n' "$key"
        return 0
    fi

    printf '%s: ' "$label"
    new_value="$(read_prompt_value "$secret")"

    if [ -z "$new_value" ]; then
        printf 'Skipped %s\n' "$key"
        return 0
    fi

    set_env_value "$key" "$new_value"
    printf 'Saved %s in %s\n' "$key" "$ENV_FILE"
}

warn_unfilled_values() {
    missing=""

    for key in TELEGRAM_BOT_TOKEN OPENAI_API_KEY OPENAI_BASE_URL; do
        current_value="$(env_value "$key")"
        if is_unfilled_value "$key" "$current_value"; then
            missing="${missing}${missing:+, }${key}"
        fi
    done

    if [ -n "$missing" ]; then
        printf 'Warning: still unfilled: %s\n' "$missing" >&2
        printf 'The service may fail until these values are set in %s.\n' "$ENV_FILE" >&2
    fi
}

select_run_method() {
    while :; do
        printf 'How do you want to run the bot? [docker/systemd] (default: docker): ' >&2
        IFS= read -r run_method || run_method=""
        normalized_method="$(printf '%s' "$run_method" | tr '[:upper:]' '[:lower:]')"

        case "$normalized_method" in
            ""|"1"|"d"|"docker")
                printf 'docker'
                return 0
                ;;
            "2"|"s"|"systemd")
                printf 'systemd'
                return 0
                ;;
            *)
                printf 'Please enter docker or systemd.\n' >&2
                ;;
        esac
    done
}

detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_STYLE="plugin"
        return 0
    fi

    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_STYLE="standalone"
        return 0
    fi

    printf 'Docker Compose is not available. Install docker compose or docker-compose.\n' >&2
    exit 1
}

run_compose() {
    if [ "$COMPOSE_STYLE" = "plugin" ]; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

compose_command() {
    if [ "$COMPOSE_STYLE" = "plugin" ]; then
        printf 'docker compose'
    else
        printf 'docker-compose'
    fi
}

is_service_running() {
    container_ids="$(run_compose ps -q "$SERVICE_NAME" 2>/dev/null || true)"

    for container_id in $container_ids; do
        is_running="$(docker inspect -f '{{.State.Running}}' "$container_id" 2>/dev/null || printf 'false')"
        if [ "$is_running" = "true" ]; then
            return 0
        fi
    done

    return 1
}

start_or_restart_docker_service() {
    if is_service_running; then
        printf 'Service %s is already running; restarting...\n' "$SERVICE_NAME"
        run_compose restart "$SERVICE_NAME"
    else
        printf 'Starting service %s...\n' "$SERVICE_NAME"
        run_compose up -d --build "$SERVICE_NAME"
    fi

    printf 'Logs command:\n  %s logs -f %s\n' "$(compose_command)" "$SERVICE_NAME"
}

detect_systemd() {
    if ! command -v systemctl >/dev/null 2>&1; then
        printf 'systemctl is not available. Install systemd or choose docker.\n' >&2
        exit 1
    fi

    if [ "$(id -u)" != "0" ] && ! command -v sudo >/dev/null 2>&1; then
        printf 'sudo is required to manage a systemd system service as a non-root user.\n' >&2
        exit 1
    fi
}

validate_systemd_service_name() {
    service_name="$1"

    case "$service_name" in
        ""|*/*)
            printf 'Invalid systemd service name: %s\n' "$service_name" >&2
            exit 1
            ;;
    esac
}

run_systemctl() {
    if [ "$(id -u)" = "0" ]; then
        systemctl "$@"
    else
        sudo systemctl "$@"
    fi
}

systemd_unit_name() {
    service_name="$1"

    case "$service_name" in
        *.service)
            printf '%s' "$service_name"
            ;;
        *)
            printf '%s.service' "$service_name"
            ;;
    esac
}

systemd_unit_path() {
    service_name="$1"
    unit_name="$(systemd_unit_name "$service_name")"

    printf '%s/%s' "$SYSTEMD_UNIT_DIR" "$unit_name"
}

systemd_service_name() {
    configured_name="$(env_value "SYSTEMD_SERVICE_NAME")"

    if is_unfilled_value "SYSTEMD_SERVICE_NAME" "$configured_name"; then
        printf '%s' "$DEFAULT_SYSTEMD_SERVICE_NAME"
    else
        printf '%s' "$configured_name"
    fi
}

systemd_service_user() {
    if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
        printf '%s' "$SUDO_USER"
    else
        id -un
    fi
}

install_systemd_unit() {
    service_name="$1"
    unit_name="$(systemd_unit_name "$service_name")"
    unit_path="$(systemd_unit_path "$service_name")"
    service_user="$(systemd_service_user)"
    tmp_file="$(mktemp)"

    {
        printf '[Unit]\n'
        printf 'Description=ChatGPT Telegram Bot\n'
        printf 'After=network-online.target\n'
        printf 'Wants=network-online.target\n'
        printf '\n'
        printf '[Service]\n'
        printf 'Type=simple\n'
        printf 'WorkingDirectory=%s\n' "$SCRIPT_DIR"
        printf 'EnvironmentFile=%s\n' "$ENV_FILE"
        printf 'ExecStart=%s -m bot\n' "$PYTHON_BIN"
        printf 'Restart=always\n'
        printf 'RestartSec=5\n'
        printf 'User=%s\n' "$service_user"
        printf '\n'
        printf '[Install]\n'
        printf 'WantedBy=multi-user.target\n'
    } > "$tmp_file"

    if [ "$(id -u)" = "0" ]; then
        install -m 0644 "$tmp_file" "$unit_path"
    else
        sudo install -m 0644 "$tmp_file" "$unit_path"
    fi

    rm -f "$tmp_file"
    run_systemctl daemon-reload
    run_systemctl enable "$unit_name"
}

ensure_systemd_unit() {
    service_name="$1"
    unit_name="$(systemd_unit_name "$service_name")"
    unit_path="$(systemd_unit_path "$service_name")"

    if systemctl cat "$unit_name" >/dev/null 2>&1 || [ -f "$unit_path" ]; then
        printf 'Using existing systemd unit %s\n' "$unit_name"
        return 0
    fi

    if [ ! -x "$PYTHON_BIN" ]; then
        printf 'Missing executable Python at %s. Create .venv before using systemd mode.\n' "$PYTHON_BIN" >&2
        exit 1
    fi

    printf 'Creating systemd unit %s\n' "$unit_path"
    install_systemd_unit "$service_name"
}

systemd_logs_command() {
    unit_name="$1"

    if [ "$(id -u)" = "0" ]; then
        printf 'journalctl -u %s -f' "$unit_name"
    else
        printf 'sudo journalctl -u %s -f' "$unit_name"
    fi
}

start_or_restart_systemd_service() {
    service_name="$(systemd_service_name)"
    validate_systemd_service_name "$service_name"
    unit_name="$(systemd_unit_name "$service_name")"
    ensure_systemd_unit "$service_name"

    if systemctl is-active --quiet "$unit_name"; then
        printf 'Systemd service %s is already running; restarting...\n' "$unit_name"
        run_systemctl restart "$unit_name"
    else
        printf 'Starting systemd service %s...\n' "$unit_name"
        run_systemctl start "$unit_name"
    fi

    printf 'Logs command:\n  %s\n' "$(systemd_logs_command "$unit_name")"
}

start_selected_service() {
    run_method="$1"

    case "$run_method" in
        "docker")
            detect_compose
            start_or_restart_docker_service
            ;;
        "systemd")
            detect_systemd
            start_or_restart_systemd_service
            ;;
    esac
}

main() {
    copy_env_example
    prompt_env_value "TELEGRAM_BOT_TOKEN" "Enter TELEGRAM_BOT_TOKEN (empty to skip)" "true"
    prompt_env_value "OPENAI_API_KEY" "Enter OPENAI_API_KEY (empty to skip)" "true"
    prompt_env_value "OPENAI_BASE_URL" "Enter OPENAI_BASE_URL (empty to skip)" "false"
    warn_unfilled_values
    run_method="$(select_run_method)"
    start_selected_service "$run_method"
}

main "$@"
