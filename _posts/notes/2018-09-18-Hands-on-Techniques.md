---
layout: post
title: "Hands-on Techniques for Computer Manipulation: Recipes"
author: Guanlin Li
tag: notes
---

## Linux

- Look up Linux Release Version
```bash
cat /etc/*release
lsb_release -a
```
- Look up the disk space of the current directory
```bash
du -h --max-depth=1 $dir_path
df -h  # show the disk usage of the whole system
```


## Conda

- Rename an environment
```bash
# https://stackoverflow.com/questions/42231764/how-can-i-rename-a-conda-environment
conda create --name $new_name --clone $old_name
conda env remove --name $old_name --all
```

## Python

- Replace sub string in a str
```python
# substr must be in `s`
s.replace(substr, new_substr)
```
