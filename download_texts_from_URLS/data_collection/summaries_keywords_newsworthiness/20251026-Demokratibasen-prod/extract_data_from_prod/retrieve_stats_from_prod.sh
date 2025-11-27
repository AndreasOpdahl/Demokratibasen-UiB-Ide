SINCE_DATE="${SINCE_DATE:-2000-01-01}"
REMOTE=demokratibasen-prod
# REMOTE=demokratibasen-demo
ssh -T $REMOTE "SINCE_DATE='$SINCE_DATE' bash -se" << 'EOF'
PG_DOCKER=app-demo-db-1
DOCKER_CMD=docker
# PG_DOCKER=kommunebasen_demo-db_1
# DOCKER_CMD="sudo docker"
PG_DB=postgres
PG_USER=postgres
#
echo
echo
echo *** Retrieved documents per type, all time ***
echo 
PSQL_CMD="SELECT doc_type, COUNT(dokument_id) FROM dokument GROUP BY doc_type ORDER BY doc_type;"
$DOCKER_CMD exec $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 -c "$PSQL_CMD"
#
echo
echo
echo *** Processed documents per type, all time ***
echo 
PSQL_CMD="SELECT d.doc_type, COUNT(d.dokument_id) FROM dokument AS d NATURAL JOIN inferens AS i GROUP BY d.doc_type ORDER BY d.doc_type;"
$DOCKER_CMD exec $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 -c "$PSQL_CMD"
#
echo
echo
echo *** Retrieved documents per date, since $SINCE_DATE ***
echo 
PSQL_CMD="SELECT DATE(tidsstempel_sett) AS date_part, COUNT(dokument_id) FROM dokument WHERE DATE(tidsstempel_sett) >= '$SINCE_DATE' GROUP BY date_part ORDER BY date_part;"
$DOCKER_CMD exec $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 -c "$PSQL_CMD"
#
echo
echo
echo *** Processed documents per date, since $SINCE_DATE ***
echo 
PSQL_CMD="SELECT DATE(d.tidsstempel_sett) AS date_part, COUNT(d.dokument_id) FROM dokument AS d NATURAL JOIN inferens AS i WHERE DATE(d.tidsstempel_sett) >= '$SINCE_DATE' GROUP BY date_part ORDER BY date_part;"
$DOCKER_CMD exec $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 -c "$PSQL_CMD"
#
echo
echo
echo *** Retrieved case presentations per date, since $SINCE_DATE ***
echo 
PSQL_CMD="SELECT DATE(tidsstempel_sett) AS date_part, COUNT(dokument_id) FROM dokument WHERE doc_type = 'case_presentation' AND DATE(tidsstempel_sett) >= '$SINCE_DATE' GROUP BY date_part ORDER BY date_part;"
$DOCKER_CMD exec $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 -c "$PSQL_CMD"
#
echo
echo
echo *** Processed case presentations per date, since $SINCE_DATE ***
echo 
PSQL_CMD="SELECT DATE(d.tidsstempel_sett) AS date_part, COUNT(d.dokument_id) FROM dokument AS d NATURAL JOIN inferens AS i WHERE d.doc_type = 'case_presentation' AND DATE(d.tidsstempel_sett) >= '$SINCE_DATE' GROUP BY date_part ORDER BY date_part;"
$DOCKER_CMD exec $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 -c "$PSQL_CMD"
#
echo
echo
echo *** Municipalities with retrieved case presentations per date, since $SINCE_DATE ***
echo 
PSQL_CMD="SELECT DATE(tidsstempel_sett) AS date_part, COUNT(DISTINCT kommune) FROM dokument WHERE doc_type = 'case_presentation' AND DATE(tidsstempel_sett) >= '$SINCE_DATE' GROUP BY date_part ORDER BY date_part;"
$DOCKER_CMD exec $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 -c "$PSQL_CMD"
#
echo
echo
echo *** Municipalities with processed case presentations per date, since $SINCE_DATE ***
echo 
PSQL_CMD="SELECT DATE(d.tidsstempel_sett) AS date_part, COUNT(DISTINCT d.kommune) FROM dokument AS d NATURAL JOIN inferens AS i WHERE d.doc_type = 'case_presentation' AND DATE(d.tidsstempel_sett) >= '$SINCE_DATE' GROUP BY date_part ORDER BY date_part;"
$DOCKER_CMD exec $PG_DOCKER psql $PG_DB --user $PG_USER -v ON_ERROR_STOP=1 -c "$PSQL_CMD"
#
EOF
