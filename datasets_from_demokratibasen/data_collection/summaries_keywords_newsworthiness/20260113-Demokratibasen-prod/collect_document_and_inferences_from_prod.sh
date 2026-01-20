CSV_FILE=url_tekst_oppsummering_from_prod20260113.csv
REMOTE=demokratibasen-prod
# REMOTE=demokratibasen-demo
ssh -T $REMOTE "REMOTE_CSV_FILE='$CSV_FILE' bash -se" << 'EOF'
SINCE_DATE="${SINCE_DATE:-2000-01-01}"
PG_DOCKER=app-demo-db-1
DOCKER_CMD=docker
# PG_DOCKER=kommunebasen_demo-db_1
# DOCKER_CMD="sudo docker"
PG_DB=postgres
PG_USER=postgres
echo
echo *** On remote! ***
echo REMOTE_CSV_FILE=$REMOTE_CSV_FILE
echo 
$DOCKER_CMD exec -i $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 << PSQL
\COPY (SELECT d.dokument_id as dok_id, d.kommune as kommune, d.doc_type as dok_type, d.tittel as dok_tittel, d.url as url, i.tittel as oppsum_tittel, i.oppsummering as oppsummering, i.personer as personer, i.nokkelord as nokkelord, i.nyhetsverdi as nyhetsverdi FROM inferens i JOIN dokument d ON i.dokument_id = d.dokument_id WHERE d.dato > '$SINCE_DATE') TO '$REMOTE_CSV_FILE' WITH CSV HEADER;
PSQL
$DOCKER_CMD cp $PG_DOCKER:$REMOTE_CSV_FILE .
$DOCKER_CMD exec $PG_DOCKER rm $REMOTE_CSV_FILE 
EOF
scp $REMOTE:$CSV_FILE .
ssh $REMOTE "rm $CSV_FILE"
