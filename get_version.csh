#!/bin/csh

# This script is intended to be used by developers on TROPoe to auto increment
# the version of the package. To use it, run this script using the pre-commit git
# hook feature. You can configure this in .git/hooks/pre-commit
#

# Get the version from git describe
set VERSION = `git describe --tags `

# Get the latest tag and commits since that tag
set VERSION = ($VERSION:as/-/ /)

# Seperate the major.minor and the number of commits, then increment the commits by 1
set MAJOR = $VERSION[1]
set COMMITS = $VERSION[2]
@ COMMITS = $COMMITS + 1
set VERSION = "$MAJOR.$COMMITS"

# Add this new version to the __init__.py
sed -i "" "s/__version__ =.*/__version__ = '$VERSION'/" TROPoe.py

# Add the __init__.py to the commit
git add TROPoe.py
