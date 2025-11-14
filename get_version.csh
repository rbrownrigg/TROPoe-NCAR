#!/bin/csh

# This script is intended to be used by developers on TROPoe to auto increment
# the version of the package. To use it, run this script using the post-commit git
# hook feature. You can configure this in .git/hooks/post-commit
#

# Get the version from git describe
set VERSION = `git describe --tags `

echo "Tagging the code with $VERSION"
sed -i "" "s/TROPoe_software_version =.*/TROPoe_software_version = '$VERSION'/" version.py

git add version.py
git commit --amend --no-edit --no-verify
