declare -A src_lang_to_full=(["ibo"]="igbo" ["hau"]="hausa" ["kin"]="kinyarwanda" ["zul"]="zulu" ["wol"]="wolof" ["twi"]="twi" ["bem"]="bemba" ["fon"]="fon" ["yor"]="yoruba" ["swa"]="swahili" )
declare -A src_lang_to_pivot=(["ibo"]="english" ["hau"]="english" ["fon"]="french" ["yor"]="english" ["swa"]="english" ["kin"]="english" ["zul"]="english" ["wol"]="french" ["twi"]="english" ["bem"]="english")

for lang in fon yor swa;
do
    CUDA_VISIBLE_DEVICES=4 python3 translation_script/nllb_translate.py --queries_file queries/official_topics/$lang/test.$lang.tsv \
        --lang ${src_lang_to_full[$lang]} --output_file queries/nllb_topics_new/test.$lang.tsv

    python3  translation_script/translate_queries_gmt.py --questions_file_path queries/official_topics/$lang/test.$lang.tsv --source ${src_lang_to_full[$lang]} --pivot ${src_lang_to_pivot[$lang]} \
        --output_file_path queries/gmt_topics_new/test.$lang.tsv

    python3 translation_script/afrimt5_translate.py --queries_file queries/official_topics/$lang/test.$lang.tsv \
        --source_lang ${src_lang_to_full[$lang]} --target_lang ${src_lang_to_pivot[$lang]}  --output_file queries/afrimt5_topics/test.$lang.tsv
done
