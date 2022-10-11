for i in $(find $1 -name "*.yaml"); do # Not recommended, will break on whitespace
    	echo Starting to process config file $i
	CUDA_VISIBLE_DEVICES=1, python3  simulate.py -c $i 
done
