#!/bin/bash


# update version control information, dump results into current directory
echo "Updating version control information for this Y2_wall"

if [ -z "$(ls -A emirge)" ]; then
  echo "missing emirge top level directory. Build emirge before updating version control information"
  exit 1
fi

cd emirge
./version.sh --output-requirements=../myreqs.txt --output-conda-env=../myenv.yml
cd ..

if [ ${MIRGE_PLATFORM} ]; then
  echo "Storing version control information for ${MIRGE_PLATFORM}"
  if [ -z "$(ls -A platforms/${MIRGE_PLATFORM})" ]; then
    echo "Creating a new storage directory platforms/${MIRGE_PLATFORM}"
    mkdir platforms/${MIRGE_PLATFORM}
  fi
  mv myreqs.txt platforms/${MIRGE_PLATFORM}/.
  mv myenv.yml platforms/${MIRGE_PLATFORM}/.
else
  echo "Unknown platform. Set the environment variable MIRGE_PLATFORM for automated build and storage"
  echo "For example in bash: export MIRGE_PLATFORM=\"mac-m1\""
  echo "Version control files will be left in the source directory"
fi
