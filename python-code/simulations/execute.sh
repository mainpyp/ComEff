for i in $(find $1 -name "*.yaml"); do # Not recommended, will break on whitespace
    	echo Starting to process config file $i
	python3  simulate.py -c $i 
done
