while getopts d:o: flag
do 
    case "${flag}" in
        d) dir="$OPTARG";;
        o) outdir="$OPTARG";;
    esac
done

find ${dir} -type f -name "*.wav" -print0 | while IFS= read -r -d '' file
do
    echo "Converting: $file"
    filepath_cmd="realpath --relative-to=$dir $file"
    filepath=$(eval $filepath_cmd)
    outfile=$outdir/$filepath
    newdir=$(dirname $outfile)
    mkdir -p $newdir
    sox $file -c 1 -b 16 -r 16000 $outfile
    echo "Converted to: $outfile"
done