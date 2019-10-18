#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 the HERA Collaboration
# Licensed under the MIT License

"""Check if there are any sessions on this Librarian that need to have
the nightly data quality procesing done. If so, launch a job to do
the processing.

Because this script is a cronjob, it should not produce any output unless
something bad happens, to avoid annoying daily emails.

"""

import os.path
import subprocess
import sys
from hera_librarian import LibrarianClient
import numpy as np

connection_name = 'local'

# Search for sessions that are:
#
# 1. Unprocessed, as evidenced by the fact that none of its associated
#    files have a 'nightlynb.processed' event
# 2. Either:
#       a. Standard session that is fully or almost fully uploaded, as
#          evidenced by there being lots of files on the Librarian.
#    or
#       b. Older than a few days, suggesting that we should just go
#          ahead and process whatever we've got.
#
#needs to be run w/ ipython: ipython cronjob.py

search = '''
{
   "no-file-has-event": "nightlynb.processed",
   "age-less-than": 14
}'''

def main():
    # connect to librarian
    cl = LibrarianClient(connection_name)

    # search for unprocessed sessions
    
    sessions = cl.search_sessions(search)['results']
    
    print("sessions", sessions)

    if not len(sessions):
        return # Nothing to do.

    # get path to plots dir
    plots_dir = os.path.dirname(sys.argv[0])
    plot_script = os.path.join(plots_dir, 'run_notebook.sh')
    
    
    # check these sessid aren't in the processed_sessid.txt file
    processed_sessid = np.loadtxt(os.path.join(plots_dir, 'processed_sessid.txt'), dtype=np.int)

    print("checking proccessed sessid")

    # filter out sessions already processed
    unprocessed_sessions = []
    for sess in sessions:
        if sess['id'] not in processed_sessid:
            unprocessed_sessions.append(sess)

    # Just pick one to process and submit the job that will
    # actually crunch it.
    sessid = unprocessed_sessions[0]['id']
    print("I'm doing something", sessid)

    env = dict(os.environ)
    env['sessid'] = str(sessid)

    subprocess.check_call(
        ['/opt/services/torque/bin/qsub', '-z', '-j', 'oe', '-o', '/lustre/aoc/projects/hera/lberkhou/H3C_plots/qsub.log', '-V', '-q', 'hera', plot_script],
        shell = False,
        env = env
    )
    print("I'm at the end")

if __name__ == '__main__':
    main()
