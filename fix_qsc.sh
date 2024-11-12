#!/bin/bash
spaces_prior=$(grep -o -m 1 'ZBS( *-*[0-9]*,[0-9]*) = *\-' $1 | wc -c)
spaces_post=$(grep -o -m 1 'ZBS( *-*[0-9]*,[0-9]*) = *\-' $1 | sed 's/[ ][ ]*/ /g' | wc -c)
extra_spaces=$((spaces_prior-spaces_post))
for input in "$@"
do
	sed -i 's/\(ZBS( *-*[0-9]*,[0-9]*) =\)[ ][ ]*/\1 /g' $input
	sed -i 's/\(ZBS( *-*[0-9]*,[0-9]*) =\) \([0-9]\)/\1 q\2/g' $input
	sed -i 's/\(ZBS( *-*[0-9]*,[0-9]*) = *\)+/\1q/g' $input
	sed -i 's/\(ZBS( *-*[0-9]*,[0-9]*) = *\)-/\1+/g' $input
	sed -i 's/\(ZBS( *-*[0-9]*,[0-9]*) = *\)q/\1-/g' $input
	for i in $(seq 1 $extra_spaces)
	do
		sed -i "s/\(ZBS( *-*[0-9]*,[0-9]*) =\) /\1  /g" $input
	done
	sed -i 's/\(RBS( *-*[0-9]*,[0-9]*) =\)[ ][ ]*/\1 /g' $input
	sed -i 's/\(RBS( *-*[0-9]*,[0-9]*) =\) \([0-9]\)/\1 q\2/g' $input
	sed -i 's/\(RBS( *-*[0-9]*,[0-9]*) = *\)+/\1q/g' $input
	sed -i 's/\(RBS( *-*[0-9]*,[0-9]*) = *\)-/\1+/g' $input
	sed -i 's/\(RBS( *-*[0-9]*,[0-9]*) = *\)q/\1-/g' $input
	for i in $(seq 1 $extra_spaces)
	do
		sed -i "s/\(RBS( *-*[0-9]*,[0-9]*) =\) /\1  /g" $input
	done
done