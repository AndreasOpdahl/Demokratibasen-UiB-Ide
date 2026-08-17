# To run, get OPENAI_API_KEY from a running Demokratibasen instance, for example ``demokratibasen-prod`` .
# Place it in the .env file (and add .env to .gitignore)
source .env

# Get date
DATE=$(date +%Y%m%d)
LOCAL_FOLDER=${DATE}-Demokratibasen-prod

# Create folder
mkdir -p ${LOCAL_FOLDER}/batch-files-${DATE}
cd ${LOCAL_FOLDER}/batch-files-${DATE}

# Retrieve all batch files from last 30 days
curl https://api.openai.com/v1/files -H "Authorization: Bearer ${OPENAI_API_KEY}"   -H "Content-Type: application/json" > file_list

# Pick batch input and output files
jq ".data[] | select(.purpose == \"batch\") | .id" file_list | sed "s/\"//g" > input_batch_file_ids
jq ".data[] | select(.purpose == \"batch_output\") | .id" file_list | sed "s/\"//g" > output_batch_file_ids


# Retrieve input file contents (the texts)
mkdir -p input_files
for file_id in $(cat input_batch_file_ids); do curl https://api.openai.com/v1/files/${file_id}/content -H "Authorization: Bearer ${OPENAI_API_KEY}" -H "Content-Type: application/json" > input_files/${file_id}.json; done

# Retrieve output file contents (the summaries - currently not used)
mkdir -p output_files
for file_id in $(cat output_batch_file_ids); do curl https://api.openai.com/v1/files/${file_id}/content -H "Authorization: Bearer ${OPENAI_API_KEY}" -H "Content-Type: application/json" > output_files/${file_id}.json; done

cd ../..
