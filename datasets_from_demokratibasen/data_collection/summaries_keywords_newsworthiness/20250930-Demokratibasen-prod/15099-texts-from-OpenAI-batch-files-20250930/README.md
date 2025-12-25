## Contents

OpenAI batch files ("30 last days") with full texts retrieved 2025-09-31. All or most correspond to demokratibasen-prod documents.

### Commands

To run, get OPENAI_API_KEY from a running Demokratibasen instance, for example ``demokratibasen-prod`` .

Retrieve all files (batch files from last 30 days):

```
curl https://api.openai.com/v1/files -H "Authorization: Bearer ${OPENAI_API_KEY}"   -H "Content-Type: application/json" > file_list
```

Pick batch input files

```
jq ".data[] | select(.purpose == \"batch\") | .id" file_list | sed "s/\"//g" > input_batch_file_ids
```

Retrieve input file contents

```
mkdir input_files
for file_id in $(cat input_batch_file_ids); do curl https://api.openai.com/v1/files/${file_id}/content -H "Authorization: Bearer ${OPENAI_API_KEY}" -H "Content-Type: application/json" > input_files/${file_id}.json; done
```

Pick batch output files

```
jq ".data[] | select(.purpose == \"batch_output\") | .id" file_list | sed "s/\"//g" > output_batch_file_ids
```

Retrieve output file contents

```
mkdir output_files
for file_id in $(cat output_batch_file_ids); do curl https://api.openai.com/v1/files/${file_id}/content -H "Authorization: Bearer ${OPENAI_API_KEY}" -H "Content-Type: application/json" > output_files/${file_id}.json; done
```
