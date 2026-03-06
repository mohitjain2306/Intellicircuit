#!/bin/bash

# IntelliCircuit Deployment Script
# Usage: bash scripts/deploy.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${BLUE}========== $1 ==========${NC}"; }

header "Checking Requirements"
command -v git     >/dev/null 2>&1 || error "Git is not installed"
command -v python3 >/dev/null 2>&1 || error "Python3 is not installed"
command -v pip     >/dev/null 2>&1 || error "pip is not installed"
log "All requirements satisfied"

header "Checking Project Structure"
if [ ! -f "backend/app.py" ]; then
    error "backend/app.py not found. Run this from the project root."
fi
if [ ! -f "backend/requirements.txt" ]; then
    error "backend/requirements.txt not found."
fi
log "Project structure looks good"

header "Installing Dependencies"
pip install -r backend/requirements.txt -q
log "Dependencies installed"

header "Running Checks"
python3 -c "import flask; print('Flask version:', flask.__version__)" || error "Flask import failed"
python3 -c "import numpy; print('Numpy version:', numpy.__version__)" || error "Numpy import failed"
log "All checks passed"

header "Git Status"
BRANCH=$(git branch --show-current)
log "Current branch: $BRANCH"

if [ -n "$(git status --porcelain)" ]; then
    warn "You have uncommitted changes:"
    git status --short
    echo ""
    read -p "Commit and push? (y/n): " choice
    if [ "$choice" = "y" ]; then
        read -p "Enter commit message: " msg
        git add .
        git commit -m "$msg"
        git push origin $BRANCH
        log "Changes pushed to GitHub"
    else
        warn "Skipping push"
    fi
else
    log "Nothing to commit, pushing latest..."
    git push origin $BRANCH
    log "Pushed to GitHub — Render will auto-deploy"
fi

header "Done"
log "Check your app at: https://intellicircuit.onrender.com"
log "Check Render logs at: https://dashboard.render.com"