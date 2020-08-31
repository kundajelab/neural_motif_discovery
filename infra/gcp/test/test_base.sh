for i in `seq 10 -1 1`
do
	echo $i
done

echo "Python version:"
python --version

echo "Checking gkmexplain"
/opt/lsgkm/src/gkmexplain
