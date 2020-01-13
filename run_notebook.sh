#!/bin/bash

librarian_conn_name=local

source activate nightlynb

echo Date: $(date)
echo sessid=$sessid

# print out help statement
if [ "$1" = '-h' -o "$1" = '--help' ] ; then
    echo 'Usage:'
    echo 'export sessid=<#3sessid>'
    echo 'qsub -V -q hera run_notebook.sh'
    exit 0
fi

if [ -z "$sessid" ] ; then
    echo "environ variable 'sessid' is undefined"
    exit 1
fi

# Exit with an error if any sub-command fails.
set -e

# Create a temporary Lustre directory for exporting the data and command the
# Librarian to populate it.

staging_dir=$(mktemp -d --tmpdir=/lustre/aoc/projects/hera/lberkhou/H3C_plots/nightlynb sessid$sessid.XXXXXX)
chmod ug+rwx "$staging_dir"

remove_staging_notes () {
    stage_files=($(ls $1))
    for f in ${stage_files[@]}
    do
        if [[ $f == *"STAG"* ]]
        then
            rm -f $staging_dir/$f
        fi
    done
}

remove_staging_notes $staging_dir
search="{\"session-id-is-exactly\": $sessid, \"name-matches\": \"%.uvh5\"}"
librarian stage-files --wait $librarian_conn_name "$staging_dir" "$search"

#remove_staging_notes $staging_dir
#search="{\"session-id-is-exactly\": $sessid, \"name-matches\": \"%.diff.uvh5\"}"
#librarian stage-files --wait $librarian_conn_name "$staging_dir" "$search"

remove_staging_notes $staging_dir
search="{\"session-id-is-exactly\": $sessid, \"name-matches\": \"%.json\"}"
librarian stage-files --wait $librarian_conn_name "$staging_dir" "$search"

remove_staging_notes $staging_dir
search="{\"session-id-is-exactly\": $sessid, \"name-matches\": \"%.calfits\"}"
librarian stage-files --wait $librarian_conn_name "$staging_dir" "$search"

remove_staging_notes $staging_dir
search="{\"session-id-is-exactly\": $sessid, \"name-matches\": \"%.flag_summary.npz\"}"
librarian stage-files --wait $librarian_conn_name "$staging_dir" "$search"

DATA_PATH=

for item in "$staging_dir"/2* ; do
    if [ -n "$DATA_PATH" ] ; then
        echo >&1 "WARNING: multiple subdirectories staged? $DATA_PATH, $item"
        exit 1
    fi
    if [ "$(basename $item)" == "2*" ] ; then
        echo >&1 "WARNING: no subdirectory staged: $item"
        exit 1
    fi
    export DATA_PATH="$item"
done

jd=$(basename $DATA_PATH)

# get more env vars
BASENBDIR=/lustre/aoc/projects/hera/lberkhou/H3C_plots
OUTPUT=data_inspect_"$jd".ipynb
OUTPUTDIR=/lustre/aoc/projects/hera/lberkhou/H3C_plots

# copy and run notebook
echo "starting notebook execution..."

jupyter nbconvert --output=$OUTPUTDIR/$OUTPUT \
  --to notebook \
  --ExecutePreprocessor.allow_errors=True \
  --ExecutePreprocessor.timeout=-1 \
  --execute $BASENBDIR/data_inspect_H3C.ipynb

echo "finished notebook execution..."

#cd to git repo
cd $OUTPUTDIR

# add to git repo
#echo "adding to GitHub repo"
#git add $OUTPUT

# add sessid to processed_sessid.txt file
echo $sessid >> $OUTPUTDIR/processed_sessid.txt
#git add $OUTPUTDIR/processed_sessid.txt

# commit and push
#git commit -m "data inspect notebook for $jd"
#git pull
#git push

# mark these files as processed (see cronjob.py). We only need to mark one
# file but we do all of the UV files since that seems like potentially handy
# information to have.

echo "adding Librarian file events"
now_unix=$(date +%s)

#for uv in $staging_dir/*/*.uvh5 ; do
#    librarian add-file-event $librarian_conn_name $uv nightlynb.processed when=$now_unix
#done

#echo "sending email to heraops"
#sed -e 's/@@JD@@/'$jd'/g' < mail_template.txt > mail.txt
#sendmail -vt < mail.txt

echo "removing staging dir"
rm -rf "$staging_dir"

echo "finished run_notebook.sh"
echo "Date:" $(date)
exit 0
