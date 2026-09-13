#!/usr/bin/env bash
# ==============================================================================
# OpenSearch Index Initialization Script
#
# Creates the 'insurance_documents' index with the k-NN vector mapping.
# Handles cluster readiness checks, path resolution, and optional recreation.
# ==============================================================================

set -euo pipefail

# Determine script directory to ensure relative paths always work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Configuration defaults (can be overridden via environment variables or CLI flags)
OPENSEARCH_URL="${OPENSEARCH_URL:-http://localhost:9200}"
INDEX_NAME="${INDEX_NAME:-insurance_documents}"
MAPPING_FILE="${MAPPING_FILE:-${SCRIPT_DIR}/index.json}"
MAX_RETRIES="${MAX_RETRIES:-30}"
RETRY_INTERVAL="${RETRY_INTERVAL:-2}"
RECREATE=false

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help message
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Initializes the OpenSearch index for the Insurance Support Agent.

Options:
  -u, --url <url>         OpenSearch endpoint URL (default: http://localhost:9200)
  -i, --index <name>      Index name to create (default: insurance_documents)
  -f, --file <path>       Path to mapping JSON file (default: infra/opensearch/index.json)
  -r, --recreate          Delete and recreate the index if it already exists
  --retries <count>       Maximum readiness check retries (default: 30)
  --interval <seconds>    Seconds between retries (default: 2)
  -h, --help              Show this help message and exit

Environment Variables:
  OPENSEARCH_URL          Override default OpenSearch endpoint
  INDEX_NAME              Override default index name
  MAPPING_FILE            Override default mapping JSON path
EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)
            OPENSEARCH_URL="$2"
            shift 2
            ;;
        -i|--index)
            INDEX_NAME="$2"
            shift 2
            ;;
        -f|--file)
            MAPPING_FILE="$2"
            shift 2
            ;;
        -r|--recreate)
            RECREATE=true
            shift
            ;;
        --retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --interval)
            RETRY_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown option: $1${NC}" >&2
            usage
            ;;
    esac
done

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}        OpenSearch Index Initializer: ${INDEX_NAME}${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "Endpoint:     ${YELLOW}${OPENSEARCH_URL}${NC}"
echo -e "Index Name:   ${YELLOW}${INDEX_NAME}${NC}"
echo -e "Mapping File: ${YELLOW}${MAPPING_FILE}${NC}"
echo ""

# 1. Verify mapping file exists
if [[ ! -f "${MAPPING_FILE}" ]]; then
    echo -e "${RED}[ERROR] Mapping file not found at: ${MAPPING_FILE}${NC}" >&2
    exit 1
fi

# 2. Wait for OpenSearch cluster readiness
echo -e "${BLUE}[1/3] Checking OpenSearch cluster availability...${NC}"
attempt=1
while true; do
    if curl -s -f --max-time 5 "${OPENSEARCH_URL}/_cluster/health" >/dev/null 2>&1; then
        cluster_info=$(curl -s --max-time 5 "${OPENSEARCH_URL}/_cluster/health")
        cluster_status=$(echo "${cluster_info}" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "reachable")
        echo -e "${GREEN}  ✓ OpenSearch is online (Cluster status: ${cluster_status})${NC}"
        break
    fi

    if [[ ${attempt} -ge ${MAX_RETRIES} ]]; then
        echo -e "${RED}[ERROR] OpenSearch at ${OPENSEARCH_URL} was unreachable after ${MAX_RETRIES} attempts.${NC}" >&2
        echo -e "${YELLOW}Ensure the container is running: docker compose up -d${NC}" >&2
        exit 1
    fi

    echo -e "  Waiting for OpenSearch (${attempt}/${MAX_RETRIES})... retrying in ${RETRY_INTERVAL}s"
    sleep "${RETRY_INTERVAL}"
    attempt=$((attempt + 1))
done

# 3. Check if index already exists
echo -e "\n${BLUE}[2/3] Checking existing index status...${NC}"
http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${OPENSEARCH_URL}/${INDEX_NAME}")

if [[ "${http_code}" == "200" ]]; then
    if [[ "${RECREATE}" == "true" ]]; then
        echo -e "${YELLOW}  Index '${INDEX_NAME}' already exists. Recreating as requested (--recreate)...${NC}"
        del_resp=$(curl -s -X DELETE --max-time 10 "${OPENSEARCH_URL}/${INDEX_NAME}")
        echo -e "  ✓ Deleted existing index: ${del_resp}"
    else
        echo -e "${GREEN}  ✓ Index '${INDEX_NAME}' already exists.${NC}"
        echo -e "${YELLOW}  No action needed. Use --recreate to wipe and re-initialize.${NC}"
        
        # Display current doc count
        doc_count=$(curl -s --max-time 5 "${OPENSEARCH_URL}/${INDEX_NAME}/_count" | grep -o '"count":[0-9]*' | cut -d: -f2 || echo "0")
        echo -e "  Current document count: ${YELLOW}${doc_count}${NC}"
        echo -e "${GREEN}================================================================${NC}"
        exit 0
    fi
elif [[ "${http_code}" == "404" ]]; then
    echo -e "  Index '${INDEX_NAME}' does not exist yet. Proceeding to create."
else
    echo -e "${YELLOW}  Unexpected status code checking index: ${http_code}${NC}"
fi

# 4. Create the index using mapping JSON
echo -e "\n${BLUE}[3/3] Creating index '${INDEX_NAME}' with k-NN vector schema...${NC}"
tmp_resp=$(mktemp)
status=$(curl -s -o "${tmp_resp}" -w "%{http_code}" --max-time 30 -X PUT "${OPENSEARCH_URL}/${INDEX_NAME}" \
    -H "Content-Type: application/json" \
    --data-binary @"${MAPPING_FILE}")
body=$(cat "${tmp_resp}")
rm -f "${tmp_resp}"

if [[ "${status}" == "200" || "${status}" == "201" ]]; then
    echo -e "${GREEN}  ✓ Index '${INDEX_NAME}' successfully created!${NC}"
    echo -e "  Response: ${body}"
else
    echo -e "${RED}[ERROR] Failed to create index. HTTP ${status}${NC}" >&2
    echo -e "Response: ${body}" >&2
    exit 1
fi

# 5. Verification
echo -e "\n${BLUE}Verification:${NC}"
curl -s --max-time 5 "${OPENSEARCH_URL}/_cat/indices/${INDEX_NAME}?v"

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}SUCCESS: Index '${INDEX_NAME}' is ready for vector search!${NC}"
echo -e "${GREEN}================================================================${NC}"
