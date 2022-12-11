python3 -m spacy download en_core_web_lg
python3 -m spacy download fr_core_news_lg

wiki_dir=$1
if [ -z $wiki_dir ]; then
	echo "Requried to provide directory path to store the Wikipedia dataset"
	echo "Example: bash scripts/generate_process_dumps.sh /path/to/wiki_dir"
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

    	bz_name="${lang}wiki-${date}-pages-articles-multistream.xml.bz2"
	bz_fn="${wiki_dir}/${bz_name}"

	wiki_json="${wiki_dir}/${lang}wiki-${date}-json"
	mkdir -p ${wiki_json}

    	if [ ! -f ${wiki_json} ]; then
		echo "preparing ${wiki_json}"

		if [ ! -f $bz_fn ]; then
		    wget "https://archive.org/download/${lang}wiki-${date}/${bz_name}" -P ${wiki_dir}
		fi

		# Extract downloaded wikipedia dumps into jsonlines 
		python3 wikiextractor/WikiExtractor.py ${wiki_dir}/${lang}wiki-${date}-pages-articles-multistream.xml.bz2 --filter_disambig_pages --json -o ${wiki_json} -s
	fi

	# Create sqlite database containing article text and titles
	python3 preprocess/retriever/build_db.py ${wiki_json} ${wiki_dir}/${lang}wiki-${date}.db
	# Generate DPR 100 words context file
	python3 preprocess/retriever/wikipedia_generate_context_tsv.py --db_path ${wiki_dir}/${lang}wiki-${date}.db --output_path_100w  ${wiki_dir}/${lang}wiki-${date}.tsv --lang ${lang}

done
