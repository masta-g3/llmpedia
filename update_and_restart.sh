#!/bin/bash

# Pull the latest changes from the git repository
git pull

# Check if there were any updates
if [ $? -eq 0 ] && [ -n "$(git diff HEAD@{1} HEAD)" ]; then
    echo "Repository updated. Restarting Docker containers..."

    # Stop all running containers
    docker stop $(docker ps -aq)

    # Remove all containers
    docker rm $(docker ps -aq)

    # Remove all unused images
    docker image prune -a -f

    # Rebuild and start the containers
    docker-compose up -d --build

    # Execute bash in the llmpedia container
    docker exec -it llmpedia_llmpedia_1 /bin/bash
else
    echo "No updates found or git pull failed."
fi
