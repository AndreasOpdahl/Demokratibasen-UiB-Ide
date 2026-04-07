# Deploying to Remote Server

This guide explains how to sync the model_test_server to a remote server using rsync.

## Prerequisites

- SSH access to the remote server
- `.pem` key file for authentication
- IP address or hostname of the remote server
- rsync installed on both local and remote machines

## Basic rsync Command

```bash
rsync -avz -e "ssh -i /path/to/your-key.pem" \
  model_test_server/ \
  user@ip-address:/path/to/destination/
```

## Command Breakdown

- `-a`: Archive mode (preserves permissions, timestamps, etc.)
- `-v`: Verbose output
- `-z`: Compress data during transfer
- `-e "ssh -i /path/to/your-key.pem"`: Use SSH with your PEM key
- `model_test_server/`: Source directory (trailing slash means contents, not the folder itself)
- `user@ip-address:/path/to/destination/`: Remote destination

## Examples

### Example 1: Sync to home directory

```bash
rsync -avz -e "ssh -i ~/.ssh/my-key.pem" \
  model_test_server/ \
  ubuntu@192.168.1.100:~/model_test_server/
```

### Example 2: Sync to specific directory

```bash
rsync -avz -e "ssh -i ~/.ssh/my-key.pem" \
  model_test_server/ \
  ubuntu@192.168.1.100:/opt/model_test_server/
```

### Example 3: Exclude certain files (like __pycache__)

```bash
rsync -avz -e "ssh -i ~/.ssh/my-key.pem" \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  model_test_server/ \
  ubuntu@192.168.1.100:~/model_test_server/
```

### Example 4: Dry run (see what would be synced)

```bash
rsync -avz -e "ssh -i ~/.ssh/my-key.pem" \
  --dry-run \
  model_test_server/ \
  ubuntu@192.168.1.100:~/model_test_server/
```

## Setting Up on Remote Server

After syncing, SSH into the remote server and set up:

```bash
# SSH into the server
ssh -i /path/to/your-key.pem user@ip-address

# Navigate to the directory
cd ~/model_test_server  # or wherever you synced it

# Install dependencies
pip install -r requirements.txt

# Make sure the adapter directory path exists or update it
# The adapter directory should be synced separately or already exist on the server
```

## Syncing Adapter Directories

If you need to sync the model adapter directory as well:

```bash
# Sync adapter directory (this might be large, so use compression)
rsync -avz -e "ssh -i ~/.ssh/my-key.pem" \
  --progress \
  model_fine_tuning_olivia/models/gemma-2-9b-apptainer-fsdp/checkpoint-5000/ \
  ubuntu@192.168.1.100:~/adapters/checkpoint-5000/
```

## Security Notes

1. **Protect your PEM key**: Set proper permissions
   ```bash
   chmod 400 /path/to/your-key.pem
   ```

2. **Use SSH config**: Create `~/.ssh/config` for easier access
   ```
   Host myserver
       HostName 192.168.1.100
       User ubuntu
       IdentityFile ~/.ssh/my-key.pem
   ```
   
   Then use:
   ```bash
   rsync -avz model_test_server/ myserver:~/model_test_server/
   ```

3. **Firewall**: Ensure the port (default 8000) is open on the remote server

## Troubleshooting

### Permission denied
- Check PEM key permissions: `chmod 400 your-key.pem`
- Verify the key is correct for the server

### Connection timeout
- Check if the IP address is correct
- Verify the server is running and accessible
- Check firewall settings

### Destination path doesn't exist
- Create the directory first: `ssh -i key.pem user@ip "mkdir -p /path/to/destination"`
