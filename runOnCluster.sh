#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage $0 <file_path>"
  exit 1
fi

FILE_PATH="$1"
REMOTE_PATH="qlora-implementation/$(basename "$FILE_PATH")"

scp -i ~/.ssh/id_ed25519_lhennecon.key -P5022 "$FILE_PATH" lhennecon@ssh.lamsade.dauphine.fr:qlora-implementation/fineTune.py

expect <<'EOF'
spawn ssh -i ~/.ssh/id_ed25519_lhennecon.key -p 5022 lhennecon@ssh.lamsade.dauphine.fr
expect "Choose your horse :"
send "3\r"
expect "$ "
send "python3 qlora-implementation/fineTune.py"
sleep 1
expect "$ "
interact
EOF
