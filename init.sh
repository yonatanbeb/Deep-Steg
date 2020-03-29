#!/bin/bash

alias sign_in=". ./sign_in.sh"
alias sign_out=". ./sign_out.sh"
alias create_profile=". ./create_profile.sh"
alias grab="python grab.py"
alias query="python query.py"
alias view="python view.py"

sign_out
echo {} > ./user_data/profiles.json
echo "\"No Clearance\"" > ./user_data/current_clearance_level.json