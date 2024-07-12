#!/usr/bin/env bash

clear

echo "Creating datasets for n clients:"

osascript <<EOF
tell application "Terminal"
    do script "python3 data/federated_data_extractor.py"
end tell
EOF

sleep 3

echo "Start federated learning on n clients:"
osascript <<EOF
tell application "Terminal"
    do script "python3 miner.py -g 1 -l 2"
end tell
EOF

sleep 3

for i in `seq 0 1`;
        do
            echo "Start client $i"
            osascript <<EOF
            tell application "Terminal"
                do script "python3 client.py -d 'data/federated_data_$i.d' -e 1"
            end tell
EOF
done

sleep 3
osascript <<EOF
tell application "Terminal"
    do script "python3 create_csv.py"
end tell
EOF
