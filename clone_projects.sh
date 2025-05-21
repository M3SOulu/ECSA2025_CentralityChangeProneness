#!/bin/bash

git clone https://github.com/FudanSELab/train-ticket Projects/train-ticket

cp -r  Projects/train-ticket Projects/train-ticket-0.0.1
git -C Projects/train-ticket-0.0.1 checkout -f 0.0.1

cp -r  Projects/train-ticket Projects/train-ticket-0.0.2
git -C Projects/train-ticket-0.0.2 checkout -f 0.0.2

cp -r  Projects/train-ticket Projects/train-ticket-0.0.3
git -C Projects/train-ticket-0.0.3 checkout -f 0.0.3

cp -r  Projects/train-ticket Projects/train-ticket-0.0.4
git -C Projects/train-ticket-0.0.4 checkout -f 0.0.4

cp -r  Projects/train-ticket Projects/train-ticket-0.1.0
git -C Projects/train-ticket-0.1.0 checkout -f v0.1.0

cp -r  Projects/train-ticket Projects/train-ticket-0.2.0
git -C Projects/train-ticket-0.2.0 checkout -f v0.2.0

cp -r  Projects/train-ticket Projects/train-ticket-1.0.0
git -C Projects/train-ticket-1.0.0 checkout -f v1.0.0

rm -rf Projects/train-ticket