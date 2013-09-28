#!/bin/bash
# get the current X server resolution
get_screen_size() {
    xdpyinfo | awk '/dimensions:/ { print $2; exit }'
}

get_screen_size
