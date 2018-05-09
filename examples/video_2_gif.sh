HEIGHT=480

# -ss sec: seek sec into file
# -t sec : read in sec
for filename in *.mp4; do
    outfile=${filename%.*}.gif
    echo "${filename} --> ${outfile}"

    # highest quality (large file size)
#    ffmpeg -i ${filename} -ss 5.0 -t 1.0 -y -filter_complex \
#    "[0:v] fps=12,scale=w=${WIDTH}:h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1"\
#    ${outfile} #2> /dev/null

    # high quality
    ffmpeg -i ${filename} -ss 5.0 -t 10.0 -y -filter_complex "[0:v] fps=12,scale=-1:${HEIGHT},split [a][b];[a] palettegen [p];[b][p] paletteuse"\
    ${outfile} #2> /dev/null
    #ffmpeg -i ${filename} -ss 5.0 -t 5.0 -y -filter_complex "[0:v] fps=12,split [a][b];[a] palettegen [p];[b][p] paletteuse"\
done

