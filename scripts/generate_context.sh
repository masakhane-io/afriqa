"""
Adapted and modified from https://github.com/crystina-z/mrtydi
"""
wiki_dir=$1
if [ -z $wiki_dir ]; then
	echo "Requried to provide directory path to store the Wikipedia dataset"
	echo "Example: sh 0_download_and_extract.wiki.sh /path/to/wiki_dir"
	exit
fi

mkdir -p $wiki_dir

for lang in en fr
do
	if [ $lang = "en" ]; then
		date="20220501"
	else
		date="20220420"
	fi

    bz_name="${lang}wiki-$date-pages-articles-multistream.xml.bz2"
	bz_fn="${wiki_dir}/$bz_name"

    wiki_json="${wiki_dir}/${lang}wiki.$date.json"

    if [ ! -f $wiki_json ]; then
		echo "preparing $wiki_json"

        if [ ! -f $bz_fn ]; then
            wget "https://archive.org/download/${lang}wiki-$date/$bz_name" -P $wiki_dir
        fi

        python WikiExtractor.py /path/to/your/xxwiki-20190201-pages-articles-multistream.xml.bz2 --filter_disambig_pages --json -o /path/to/output/directory -s
    
    fi