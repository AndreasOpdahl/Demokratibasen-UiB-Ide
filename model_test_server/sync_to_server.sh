#!/bin/bash
# Helper script to sync model_test_server to a remote server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PEM_KEY=""
REMOTE_USER=""
REMOTE_IP=""
REMOTE_PATH="~/model_test_server"
DRY_RUN=false
SYNC_CHECKPOINT=false
CHECKPOINT_PATH=""

# Function to print usage
usage() {
    echo "Usage: $0 --pem-key KEY --user USER --ip IP [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --pem-key KEY          Path to PEM key file"
    echo "  --user USER            Remote server username"
    echo "  --ip IP                Remote server IP address or hostname"
    echo ""
    echo "Options:"
    echo "  --remote-path PATH     Remote destination path (default: ~/model_test_server)"
    echo "  --dry-run               Show what would be synced without actually syncing"
    echo "  --sync-checkpoint      Also sync the model checkpoint"
    echo "  --checkpoint-path PATH  Path to checkpoint (required if --sync-checkpoint)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --pem-key ~/.ssh/my-key.pem --user ubuntu --ip 192.168.1.100"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pem-key)
            PEM_KEY="$2"
            shift 2
            ;;
        --user)
            REMOTE_USER="$2"
            shift 2
            ;;
        --ip)
            REMOTE_IP="$2"
            shift 2
            ;;
        --remote-path)
            REMOTE_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --sync-checkpoint)
            SYNC_CHECKPOINT=true
            shift
            ;;
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$PEM_KEY" ] || [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_IP" ]; then
    echo -e "${RED}Error: --pem-key, --user, and --ip are required${NC}"
    usage
fi

# Check if PEM key exists
if [ ! -f "$PEM_KEY" ]; then
    echo -e "${RED}Error: PEM key file not found: $PEM_KEY${NC}"
    exit 1
fi

# Set proper permissions on PEM key
chmod 400 "$PEM_KEY" 2>/dev/null || echo -e "${YELLOW}Warning: Could not set permissions on PEM key${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_DIR="$SCRIPT_DIR"

# Build rsync command
RSYNC_CMD="rsync -avz -e \"ssh -i $PEM_KEY\""

if [ "$DRY_RUN" = true ]; then
    RSYNC_CMD="$RSYNC_CMD --dry-run"
    echo -e "${YELLOW}DRY RUN MODE - No files will be transferred${NC}"
fi

# Exclude patterns
EXCLUDE_PATTERNS="--exclude '__pycache__' --exclude '*.pyc' --exclude '*.pyo' --exclude '.git' --exclude '*.log'"

# Sync model_test_server
echo -e "${GREEN}Syncing model_test_server...${NC}"
echo "Source: $SOURCE_DIR/"
echo "Destination: $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH/"
echo ""

eval "$RSYNC_CMD $EXCLUDE_PATTERNS $SOURCE_DIR/ $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH/"

if [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}✓ model_test_server synced successfully!${NC}"
fi

# Sync checkpoint if requested
if [ "$SYNC_CHECKPOINT" = true ]; then
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo -e "${RED}Error: --checkpoint-path is required when using --sync-checkpoint${NC}"
        exit 1
    fi
    
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo -e "${RED}Error: Checkpoint path does not exist: $CHECKPOINT_PATH${NC}"
        exit 1
    fi
    
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")
    REMOTE_CHECKPOINT_PATH="~/checkpoints/$CHECKPOINT_NAME"
    
    echo ""
    echo -e "${GREEN}Syncing checkpoint...${NC}"
    echo "Source: $CHECKPOINT_PATH/"
    echo "Destination: $REMOTE_USER@$REMOTE_IP:$REMOTE_CHECKPOINT_PATH/"
    echo -e "${YELLOW}Note: This may take a while depending on checkpoint size...${NC}"
    echo ""
    
    eval "$RSYNC_CMD --progress $CHECKPOINT_PATH/ $REMOTE_USER@$REMOTE_IP:$REMOTE_CHECKPOINT_PATH/"
    
    if [ "$DRY_RUN" = false ]; then
        echo -e "${GREEN}✓ Checkpoint synced successfully!${NC}"
        echo ""
        echo -e "${YELLOW}Note: Update the checkpoint path in your server startup command:${NC}"
        echo "  --checkpoint_path $REMOTE_CHECKPOINT_PATH"
    fi
fi

echo ""
echo -e "${GREEN}Sync complete!${NC}"
echo ""
echo "Next steps on remote server:"
echo "  1. SSH into the server:"
echo "     ssh -i $PEM_KEY $REMOTE_USER@$REMOTE_IP"
echo ""
echo "  2. Navigate to the directory:"
echo "     cd $REMOTE_PATH"
echo ""
echo "  3. Install dependencies:"
echo "     pip install -r requirements.txt"
echo ""
if [ "$SYNC_CHECKPOINT" = true ]; then
    echo "  4. Start the server:"
    echo "     python app.py --checkpoint_path $REMOTE_CHECKPOINT_PATH --model_name gemma-2-9b"
else
    echo "  4. Start the server (update checkpoint path as needed):"
    echo "     python app.py --checkpoint_path /path/to/checkpoint --model_name gemma-2-9b"
fi
