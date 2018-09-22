---
layout: post
title: "Hands-on Techniques for Computer Manipulation"
author: Guanlin Li
tag: notes
---

## Linux

- Look up Linux Release Version
```
cat /etc/*release
lsb_release -a
```
- Look up the disk space of the current directory
```
du -h $dir_path
```
## Conda

- Rename an environment
```
# https://stackoverflow.com/questions/42231764/how-can-i-rename-a-conda-environment
conda create --name $new_name --clone $old_name
conda env remove --name $old_name --all
```
