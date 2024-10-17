#!/bin/bash
for input in "$@"
do
	sed -i 's/\(ZBS(-*[0-9]*,[0-9]*) = \)+/\1q/g' $input
	sed -i 's/\(ZBS(-*[0-9]*,[0-9]*) = \)-/\1+/g' $input
	sed -i 's/\(ZBS(-*[0-9]*,[0-9]*) = \)q/\1-/g' $input
done
